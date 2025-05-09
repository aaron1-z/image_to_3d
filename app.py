import streamlit as st
import torch
import numpy as np
import os
import io
import time
import base64
from PIL import Image
import trimesh
from scipy.spatial import Delaunay
import warnings
import cv2
import plotly.graph_objects as go

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide", page_title="Photo to 3D Converter")

# --- Configuration ---
OUTPUT_DIR = "output_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Model Loading (Cached for performance) ---
@st.cache_resource
def get_midas_model():
    try:
        # Try different model types in case one fails
        model_types = ["DPT_Large", "MiDaS_small"]
        
        for model_type in model_types:
            try:
                # Try different repo names
                try:
                    model = torch.hub.load("isl-org/MiDaS", model_type, trust_repo=True)
                except:
                    model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
                
                device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                model.to(device)
                model.eval()
                
                # Load transforms
                try:
                    transform_hub = torch.hub.load("isl-org/MiDaS", "transforms", trust_repo=True)
                except:
                    transform_hub = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
                
                if model_type == "DPT_Large":
                    transform = transform_hub.dpt_transform
                else:
                    transform = transform_hub.small_transform
                
                st.success(f"✅ MiDaS model loaded successfully ({model_type}) on {device}")
                return model, transform, device
            except Exception as model_error:
                st.warning(f"Could not load MiDaS model {model_type}: {model_error}. Trying next model...")
                continue
                
        # If we get here, all models failed
        st.error("Failed to load any MiDaS model. Please check your internet connection and PyTorch Hub access.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading MiDaS model: {e}")
        return None, None, None

@st.cache_resource
def get_rembg_session():
    try:
        import rembg
        session = rembg.new_session()
        st.success("✅ Background removal model (rembg) loaded successfully")
        return session
    except ImportError:
        st.error("Error: rembg is not installed. Please install it using 'pip install rembg[gpu]' or 'pip install rembg'")
        return None
    except Exception as e:
        st.error(f"Error loading rembg model: {e}")
        return None

# --- Helper Functions ---
def remove_background(input_image_bytes, session):
    """Remove the background from an image using rembg"""
    try:
        import rembg
        output_bytes = rembg.remove(input_image_bytes, session=session)
        return output_bytes
    except Exception as e:
        st.warning(f"Background removal failed: {e}. Using original image.")
        return input_image_bytes

def save_mesh_to_file(mesh, filename_base, file_format):
    """Save the mesh to a file in the specified format"""
    filepath = os.path.join(OUTPUT_DIR, f"{filename_base}.{file_format.lower()}")
    try:
        if file_format.lower() == "obj":
            mesh.export(filepath, file_type='obj')
        elif file_format.lower() == "stl":
            mesh.export(filepath, file_type='stl')
        elif file_format.lower() == "ply":
            mesh.export(filepath, file_type='ply')
        elif file_format.lower() == "glb":
            mesh.export(filepath, file_type='glb')
        else:
            st.error(f"Unsupported file format for saving: {file_format}")
            return None
        return filepath
    except Exception as e:
        st.error(f"Error saving mesh to {file_format}: {e}")
        return None

def enhance_depth_map(depth_map, alpha_mask=None):
    """Enhance the depth map using various CV techniques"""
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = depth_norm.astype(np.uint8)
    
    depth_filtered = cv2.bilateralFilter(depth_norm, 9, 75, 75)
    depth_eq = cv2.equalizeHist(depth_filtered)
    
    if alpha_mask is not None:
        mask = (alpha_mask > 128).astype(np.uint8) * 255
        depth_eq = cv2.bitwise_and(depth_eq, depth_eq, mask=mask)
        depth_eq[mask == 0] = 255 
    
    depth_enhanced = (depth_eq / 255.0) * (depth_map.max() - depth_map.min()) + depth_map.min()
    return depth_enhanced

def create_mesh_from_depth(depth_map_np, alpha_channel_np, img_rgb=None, scale_factor=0.1, max_points=10000):
    if depth_map_np is None or alpha_channel_np is None:
        return None
    
    enhanced_depth = enhance_depth_map(depth_map_np, alpha_channel_np)
    h, w = enhanced_depth.shape
    
    min_depth, max_depth = enhanced_depth.min(), enhanced_depth.max()
    if max_depth == min_depth: 
        normalized_depth = np.zeros_like(enhanced_depth)
    else:
        normalized_depth = (enhanced_depth - min_depth) / (max_depth - min_depth)
    
    height_map = (1 - normalized_depth) * scale_factor
    mask = alpha_channel_np > 128
    
    x = np.linspace(-w / 2 * 0.01, w / 2 * 0.01, w)
    y = np.linspace(-h / 2 * 0.01, h / 2 * 0.01, h)
    xx, yy = np.meshgrid(x, y)
    
    points_x = xx[mask]
    points_y = yy[mask]
    points_z = height_map[mask]
    
    if len(points_x) < 4: # Delaunay needs at least 4 points for 2D triangulation
        st.warning("Not enough opaque points after masking to create a 3D mesh (need at least 4).")
        return None
    
    colors_array = None
    if img_rgb is not None:
        colors_array = img_rgb[mask]
    
    if len(points_x) > max_points:
        indices = np.random.choice(len(points_x), max_points, replace=False)
        points_x = points_x[indices]
        points_y = points_y[indices]
        points_z = points_z[indices]
        if colors_array is not None:
            colors_array = colors_array[indices]
    
    points_3d = np.column_stack((points_x, points_y, points_z))
    
    try:
        points_2d = np.column_stack((points_x, points_y))
        if len(np.unique(points_2d, axis=0)) < 4:
             st.warning("Not enough unique 2D points for Delaunay triangulation. Attempting simpler mesh.")
             # Fallback for too few unique points
             if colors_array is not None:
                 cloud = trimesh.PointCloud(points_3d, colors=(colors_array / 255.0).astype(np.float32))
             else:
                 cloud = trimesh.PointCloud(points_3d)
             try: return cloud.convex_hull
             except: return None

        tri = Delaunay(points_2d)
        
        vertex_colors_normalized = None
        if colors_array is not None and len(colors_array) == len(points_3d):
            vertex_colors_normalized = (colors_array / 255.0).astype(np.float32)
            # Ensure colors are in [0,1] range and correct type
            vertex_colors_normalized = np.clip(vertex_colors_normalized, 0.0, 1.0)

        mesh = trimesh.Trimesh(
            vertices=points_3d, 
            faces=tri.simplices,
            vertex_colors=vertex_colors_normalized
        )
        
        from scipy.spatial import ConvexHull
        try:
            hull_2d = ConvexHull(points_2d) # Uses points_2d from above (potentially downsampled)
            boundary_vertex_indices_in_points_2d = hull_2d.vertices 

            # Map these indices to the original indices in points_3d (before downsampling) is tricky
            # Assuming points_2d and points_3d used here are *after* any downsampling.
            # The indices from hull_2d.vertices directly correspond to rows in points_3d/points_2d.
            
            min_z_val = points_z.min()
            thickness = max(0.01, (points_z.max() - min_z_val) * 0.05) # Reduced thickness
            
            bottom_points_3d = points_3d.copy()
            bottom_points_3d[:, 2] = min_z_val - thickness
            
            all_vertices = np.vstack([points_3d, bottom_points_3d])
            
            all_vertex_colors = None
            if vertex_colors_normalized is not None:
                bottom_colors = vertex_colors_normalized * 0.5 # Darker for bottom
                all_vertex_colors = np.vstack([vertex_colors_normalized, bottom_colors])
            
            top_faces = tri.simplices
            bottom_faces = tri.simplices + len(points_3d)
            bottom_faces = bottom_faces[:, ::-1]
            
            side_faces = []
            for i in range(len(boundary_vertex_indices_in_points_2d)):
                idx1_top = boundary_vertex_indices_in_points_2d[i]
                idx2_top = boundary_vertex_indices_in_points_2d[(i + 1) % len(boundary_vertex_indices_in_points_2d)]
                
                idx1_bottom = idx1_top + len(points_3d)
                idx2_bottom = idx2_top + len(points_3d)
                
                side_faces.append([idx1_top, idx2_top, idx2_bottom])
                side_faces.append([idx1_top, idx2_bottom, idx1_bottom])
            
            all_faces = np.vstack([top_faces, bottom_faces, np.array(side_faces)])
            
            solid_mesh = trimesh.Trimesh(
                vertices=all_vertices, 
                faces=all_faces,
                vertex_colors=all_vertex_colors
            )
            solid_mesh.process()
            
            if len(solid_mesh.faces) > 20000:
                solid_mesh = solid_mesh.simplify_quadratic_decimation(20000)
            
            return solid_mesh
        except Exception as hull_error:
            st.warning(f"Could not create solid boundary (error: {hull_error}), returning surface mesh.")
            mesh.process() # Process the surface mesh at least
            return mesh
    
    except Exception as e:
        st.warning(f"Mesh generation failed (error: {e}). Attempting convex hull from point cloud.")
        try:
            vertex_colors_normalized_fallback = None
            if colors_array is not None:
                 vertex_colors_normalized_fallback = (colors_array / 255.0).astype(np.float32)
                 vertex_colors_normalized_fallback = np.clip(vertex_colors_normalized_fallback, 0.0, 1.0)

            cloud = trimesh.PointCloud(points_3d, colors=vertex_colors_normalized_fallback)
            hull_mesh = cloud.convex_hull
            return hull_mesh
        except Exception as cloud_error:
            st.error(f"Failed to create any mesh (point cloud hull error: {cloud_error})")
            return None

def create_animated_gif(mesh, filename, duration=3, fps=20, size=(400, 400)):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not isinstance(mesh, trimesh.Trimesh) or not mesh.vertices.shape[0] > 0:
        st.warning("Cannot create GIF: Invalid mesh provided.")
        return None
    try:
        scene = mesh.scene() # Use mesh.scene() for a scene with just the mesh
        
        # Initial camera setup
        scene.camera. FIC = mesh.centroid # Focus on centroid
        scene.camera.distance = trimesh.util.diagonal(mesh.bounding_box.bounds) * 1.5 # Auto-distance
        scene.camera_transform = scene.camera.look_at(mesh.vertices) # Ensure it's looking at the mesh

        n_frames = int(duration * fps)
        frames = []
        
        # Original transform of the mesh in the scene
        node_name = list(scene.graph.nodes_geometry)[0] # Get the name of the mesh node

        for i in range(n_frames):
            angle = (i / n_frames) * 2 * np.pi 
            rotation = trimesh.transformations.rotation_matrix(
                angle=angle,
                direction=[0, 1, 0], # Rotate around Y-axis
                point=mesh.centroid # Rotate around mesh's own centroid
            )
            scene.graph[node_name] = rotation # Apply rotation transform to the mesh node
            
            try:
                data = scene.save_image(resolution=size, visible=True)
                image = Image.open(io.BytesIO(data))
                frames.append(image)
            except Exception as render_err:
                st.warning(f"Frame {i} rendering error for GIF: {render_err}")
                # Optionally add a blank frame or skip
                continue 
        
        if not frames:
            st.error("No frames generated for GIF. Rendering might have failed.")
            return None

        frames[0].save(
            filename, 
            save_all=True, 
            append_images=frames[1:], 
            duration=int(1000/fps), 
            loop=0,
            optimizer=False # Turn off optimizer if it causes issues
        )
        return filename
    except Exception as e:
        st.error(f"Could not create GIF: {e}")
        # import traceback
        # st.error(traceback.format_exc())
        return None

def plot_3d_mesh(mesh):
    vertices = mesh.vertices
    faces = mesh.faces
    
    fig_data = {
        "x": vertices[:, 0], "y": vertices[:, 1], "z": vertices[:, 2],
        "i": faces[:, 0], "j": faces[:, 1], "k": faces[:, 2],
        "opacity": 1.0,
        "lighting": dict(ambient=0.4, diffuse=0.8, specular=0.2, roughness=0.4, fresnel=0.1), # Adjusted lighting
        "lightposition": dict(x=1500, y=1500, z=2500) # Further light source
    }

    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None and len(mesh.visual.vertex_colors) > 0:
        colors_rgba = mesh.visual.vertex_colors
        if colors_rgba.dtype in [np.float32, np.float64]: # Typically in [0,1]
            colors_rgb_int = (np.clip(colors_rgba[:, :3], 0, 1) * 255).astype(np.uint8)
        elif colors_rgba.dtype == np.uint8: # Typically in [0,255]
            colors_rgb_int = colors_rgba[:, :3]
        else: # Fallback if unexpected dtype
            st.warning(f"Unexpected vertex color dtype: {colors_rgba.dtype}. Using default shading.")
            fig_data.update({"colorscale": 'Viridis', "intensity": vertices[:, 2]})
        
        if 'colorscale' not in fig_data: # if no fallback was triggered
            plotly_colors = [f'rgb({r},{g},{b})' for r, g, b in colors_rgb_int]
            fig_data["vertexcolor"] = plotly_colors
    else: 
        fig_data.update({"colorscale": 'Blues', "intensity": vertices[:, 2]}) # Use Z-height for coloring
    
    fig = go.Figure(data=[go.Mesh3d(**fig_data)])
    
    fig.update_layout(
        scene=dict(
            aspectmode='data', # Important for correct proportions
            xaxis=dict(showticklabels=False, title='', backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(128,128,128,0.2)"),
            yaxis=dict(showticklabels=False, title='', backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(128,128,128,0.2)"),
            zaxis=dict(showticklabels=False, title='', backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(128,128,128,0.2)"),
            camera_eye=dict(x=1.5, y=1.5, z=1.0) # Adjusted camera
        ),
        margin=dict(l=0, r=0, b=0, t=0), # No margins
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def add_logo_and_style():
    """Add logo and custom CSS"""
    st.markdown(
        """
        <style>
        .title-container {
            display: flex;
            align-items: center;
            margin-bottom: 1rem;
        }
        .title-container h1 {
            margin: 0;
            margin-left: 1rem;
            font-size: 2.2rem; /* Slightly larger title */
            color: #333333; /* CHANGED: Dark gray for title text for better contrast */
        }
        .logo {
            width: 50px;
            height: 50px;
        }
        .stApp {
            background-color: #e8edf1; /* CHANGED: A slightly more noticeable off-white/lightest gray-blue */
        }
        .card {
            background-color: white;
            border-radius: 8px; /* Softer corners */
            padding: 1.5rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); /* Softer shadow */
            margin-bottom: 1rem;
        }
        .download-btn { /* Custom class for HTML download links if used */
            background-color: #007bff; /* Primary blue */
            color: white;
            padding: 10px 18px; /* Adjusted padding */
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 15px;
            transition: background-color 0.2s ease; /* Smooth hover */
        }
        .download-btn:hover {
            background-color: #0056b3; /* Darker blue on hover */
        }
        .success-box {
            background-color: #d4edda;
            color: #155724; /* This is already a dark green, good contrast */
            padding: 1rem;
            border-radius: 5px;
            border-left: 5px solid #28a745; /* Success accent */
            margin-bottom: 1rem;
        }
        .info-box {
            background-color: #d1ecf1;
            color: #0c5460; /* This is already a dark teal/blue, good contrast */
            padding: 1rem;
            border-radius: 5px;
            border-left: 5px solid #17a2b8; /* Info accent */
            margin-bottom: 1rem;
        }
        /* Style Streamlit's native download button */
        div[data-testid="stDownloadButton"] > button {
            background-color: #007bff;
            color: white;
            padding: 10px 18px !important; /* Ensure padding is applied */
            border: none !important;
            border-radius: 5px !important;
            cursor: pointer !important;
            font-size: 15px !important;
            transition: background-color 0.2s ease !important;
            width: auto !important; /* Or a specific width if desired */
            display: inline-flex !important; /* Helps with alignment */
            justify-content: center;
            align-items: center;
        }
        div[data-testid="stDownloadButton"] > button:hover {
            background-color: #0056b3 !important;
            color: white !important; /* Ensure text color stays white on hover */
        }
        </style>
        <div class="title-container">
            <svg class="logo" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#007bff"> {/* Logo color is blue */}
                <path d="M0 0h24v24H0z" fill="none"/>
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-13h2v2.14c1.72.45 3 2 3 3.86 0 2.21-1.79 4-4 4s-4-1.79-4-4c0-1.86 1.28-3.41 3-3.86V7zm0 5c.28 0 .5.22.5.5s-.22.5-.5.5-.5-.22-.5-.5.22-.5.5-.5z"/>
            </svg>
            <h1>Photo to 3D Model Converter</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- UI Layout ---
add_logo_and_style()

st.markdown(
    """
    <div class="info-box">
        <p>🖼️➡️🧊 This app converts a single photo into a 3D model.
        Upload an image with a clear subject. The app will attempt to remove the background,
        estimate depth, and generate a 3D mesh.</p>
        <p>Supported output formats: OBJ, STL, PLY, GLB, and animated GIF.</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# --- Load Models ---
# Use session state to ensure models are loaded only once per session effectively
if 'models_loaded_flag' not in st.session_state:
    st.session_state.models_loaded_flag = False
    st.session_state.midas_model = None
    st.session_state.midas_transform = None
    st.session_state.midas_device = None
    st.session_state.rembg_session = None

if not st.session_state.models_loaded_flag:
    with st.spinner("Loading AI models... This might take a moment on first run."):
        st.session_state.midas_model, st.session_state.midas_transform, st.session_state.midas_device = get_midas_model()
        st.session_state.rembg_session = get_rembg_session()
        if st.session_state.midas_model is not None and st.session_state.rembg_session is not None:
            st.session_state.models_loaded_flag = True
        else:
            st.error("One or more AI models failed to load. The application might not function correctly. Please check console logs and ensure dependencies are installed.")

# --- Sidebar ---
st.sidebar.title("⚙️ Settings")

with st.sidebar.expander("🛠️ Advanced Model Settings", expanded=False):
    scale_factor = st.slider("Depth Scale Factor", 0.01, 0.3, 0.08, 0.01, 
                             help="Controls the 'extrusion' intensity. Higher values = more pronounced depth. (Default: 0.08)")
    max_points = st.slider("Mesh Detail (Max Points)", 2000, 30000, 12000, 1000,
                           help="Max points in the 3D mesh. Higher = more detail & processing time. (Default: 12000)")
    texture_quality_enabled = st.checkbox("Enable Color Texture", value=True, 
                                          help="Use original image colors for the 3D model's texture.")
    smoothing = st.slider("Depth Map Smoothing", 0, 100, 20, 5,
                          help="Applies blur to the depth map pre-mesh. Reduces noise, may soften details. (Default: 20)")

st.sidebar.markdown("---")
st.sidebar.markdown("### ✨ Tips for Best Results")
st.sidebar.markdown("""
- Use images with a **single, clear subject** and good contrast against the background.
- **Good, even lighting** with minimal harsh shadows works best.
- **Simpler object shapes** are generally easier to reconstruct.
- If background removal is poor, try an image with a less cluttered background.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 How It Works")
st.sidebar.markdown("""
1.  **Background Removal:** `rembg` library isolates the subject.
2.  **Depth Estimation:** MiDaS AI model predicts pixel depth.
3.  **Mesh Generation:** `Trimesh` & `SciPy` build a 3D mesh from the depth map, optionally textured.
4.  **Solidification:** The surface mesh is extruded for a watertight model.
""")
st.sidebar.markdown("---")
st.sidebar.info("App Version: 1.2.0 (Colors Updated)")


# --- Image Input ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📸 Upload Your Image")
uploaded_file = st.file_uploader("Choose a photo (JPG, PNG)", type=["jpg", "jpeg", "png"], key="file_uploader")
st.markdown('</div>', unsafe_allow_html=True)

# --- Main Processing ---
if 'generated_mesh' not in st.session_state:
    st.session_state.generated_mesh = None
if 'depth_map_display' not in st.session_state:
    st.session_state.depth_map_display = None
if 'mesh_filename_base' not in st.session_state:
    st.session_state.mesh_filename_base = f"model_{int(time.time())}"


if uploaded_file is not None:
    if not st.session_state.models_loaded_flag:
        st.error("Required AI models are not loaded. Please refresh or check error messages.")
    else:
        try:
            st.session_state.mesh_filename_base = f"photo_3d_{os.path.splitext(uploaded_file.name)[0]}_{int(time.time())}"
            
            progress_bar_container = st.empty()
            status_text_container = st.empty()

            with progress_bar_container: progress_bar = st.progress(0)
            with status_text_container: status_text = st.text("Initializing...")

            col1_display, col2_display = st.columns(2)
            with col1_display:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Original Image")
                uploaded_file.seek(0)
                img_pil_for_display = Image.open(uploaded_file).convert("RGB")
                st.image(img_pil_for_display, caption="Uploaded Image", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            status_text.text("⏳ Step 1/3: Removing background...")
            progress_bar.progress(20)
            
            uploaded_file.seek(0)
            input_bytes = uploaded_file.read()
            img_no_bg_bytes = remove_background(input_bytes, st.session_state.rembg_session)
            img_no_bg_pil = Image.open(io.BytesIO(img_no_bg_bytes)).convert("RGBA")
            
            with col2_display:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Background Removed")
                st.image(img_no_bg_pil, caption="Image after Background Removal", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            status_text.text("🧠 Step 2/3: Estimating depth map...")
            progress_bar.progress(50)
            
            img_rgb_np = np.array(img_no_bg_pil.convert("RGB"))
            alpha_channel_np = np.array(img_no_bg_pil)[:, :, 3]
            img_for_midas = img_rgb_np.copy() # MiDaS expects RGB
            
            input_batch = st.session_state.midas_transform(img_for_midas).to(st.session_state.midas_device)
            with torch.no_grad():
                prediction = st.session_state.midas_model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_for_midas.shape[:2],
                    mode="bicubic", align_corners=False,
                ).squeeze()
            depth_map_np_raw = prediction.cpu().numpy()
            
            if smoothing > 0:
                k_size = int((smoothing / 100.0) * 19) + 1 
                if k_size % 2 == 0: k_size +=1 
                depth_map_np_processed = cv2.GaussianBlur(depth_map_np_raw, (k_size, k_size), 0)
            else:
                depth_map_np_processed = depth_map_np_raw
            
            depth_map_display_norm = cv2.normalize(depth_map_np_processed, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_map_colored = cv2.applyColorMap(depth_map_display_norm, cv2.COLORMAP_MAGMA)
            alpha_mask_for_display = cv2.cvtColor(alpha_channel_np, cv2.COLOR_GRAY2BGR)
            alpha_mask_for_display = (alpha_mask_for_display > 128).astype(np.uint8) # Binary mask
            depth_map_masked_display = cv2.bitwise_and(depth_map_colored, depth_map_colored, mask=alpha_mask_for_display[:,:,0])
            st.session_state.depth_map_display = Image.fromarray(depth_map_masked_display)
            
            status_text.text("🧊 Step 3/3: Generating 3D model...")
            progress_bar.progress(80)
            
            rgb_for_texture = img_rgb_np.copy() if texture_quality_enabled else None
            
            st.session_state.generated_mesh = create_mesh_from_depth(
                depth_map_np_processed, alpha_channel_np,
                img_rgb=rgb_for_texture, scale_factor=scale_factor, max_points=max_points
            )
            
            progress_bar.progress(100)
            status_text_container.empty() # Clear status text
            progress_bar_container.empty() # Clear progress bar

            if st.session_state.generated_mesh is not None and isinstance(st.session_state.generated_mesh, trimesh.Trimesh):
                st.markdown(
                    f"""
                    <div class="success-box">
                        <h4>🎉 3D model generated successfully!</h4>
                        <ul>
                            <li>Vertices: {len(st.session_state.generated_mesh.vertices)}</li>
                            <li>Faces: {len(st.session_state.generated_mesh.faces)}</li>
                            <li>Watertight: {"✅ Yes" if st.session_state.generated_mesh.is_watertight else "⚠️ No (May have holes or be a surface only)"}</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("❌ Failed to generate 3D model. Try different settings or a different image.")
                st.session_state.generated_mesh = None # Ensure it's None if failed

        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
            # import traceback # Uncomment for detailed debugging
            # st.error(traceback.format_exc())
            st.session_state.generated_mesh = None # Ensure it's None on error

# --- 3D Model Display and Download ---
if st.session_state.generated_mesh is not None and isinstance(st.session_state.generated_mesh, trimesh.Trimesh):
    if len(st.session_state.generated_mesh.vertices) > 0 and len(st.session_state.generated_mesh.faces) > 0:
        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("💎 Your 3D Model")
        
        tab1, tab2, tab3 = st.tabs(["Interactive 3D View", "🔄 Animation (GIF)", "📉 Depth Map"])
        
        with tab1:
            with st.spinner("Loading interactive 3D view..."):
                fig = plot_3d_mesh(st.session_state.generated_mesh)
                st.plotly_chart(fig, use_container_width=True, config={'displaylogo': False})
        
        with tab2:
            gif_placeholder = st.empty()
            if gif_placeholder.button("Generate Animation GIF", key="gen_gif_btn", help="Click to render a rotating GIF of the model."):
                with st.spinner("🎬 Creating animated GIF... This can take some time."):
                    gif_path = os.path.join(OUTPUT_DIR, f"{st.session_state.mesh_filename_base}_animation.gif")
                    gif_created = create_animated_gif(st.session_state.generated_mesh, gif_path, duration=4, fps=15)
                    if gif_created and os.path.exists(gif_path):
                        st.image(gif_path, caption="Rotating 3D Model Animation", use_container_width=True)
                        with open(gif_path, "rb") as f_gif:
                            st.download_button(
                                label="Download Animation (GIF)", data=f_gif,
                                file_name=os.path.basename(gif_path), mime="image/gif", key="dl_gif"
                            )
                    else:
                        st.warning("Could not create animation. Mesh might be too simple or an error occurred.")
            else:
                 gif_placeholder.info("Click the button above to generate a GIF animation.")

        with tab3:
            if st.session_state.depth_map_display is not None:
                st.image(st.session_state.depth_map_display, caption="Estimated Depth Map (Colors indicate distance)", use_container_width=True)
            else:
                st.info("Depth map not generated or unavailable for this image.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("💾 Download Model Files")
        cols_download = st.columns(4)
        download_formats = {"OBJ": "model/obj", "STL": "model/stl", "PLY": "application/vnd.ply", "GLB": "model/gltf-binary"}
        
        for i, (fmt, mime_type) in enumerate(download_formats.items()):
            with cols_download[i % 4]:
                filepath = save_mesh_to_file(st.session_state.generated_mesh, st.session_state.mesh_filename_base, fmt)
                if filepath and os.path.exists(filepath):
                    with open(filepath, "rb") as fp_dl:
                        st.download_button(
                            label=f"Download .{fmt}", data=fp_dl,
                            file_name=os.path.basename(filepath), mime=mime_type, key=f"dl_{fmt.lower()}"
                        )
                else:
                    st.markdown(f".{fmt} N/A") # Should not happen if save works
        st.markdown('</div>', unsafe_allow_html=True)

    else: # Mesh object exists but has no vertices/faces
        if uploaded_file is not None: # Only show if an image was processed
             st.warning("The generated 3D model is empty (no vertices or faces). This can happen with very flat images or if processing failed to extract geometry. Try different settings or another image.")

elif uploaded_file is not None and st.session_state.generated_mesh is None:
    # This case handles when processing was attempted but generated_mesh is still None (e.g. error before assignment)
    st.info("Upload an image to get started. If processing failed, check error messages above, try different settings, or use a different image.")

st.markdown("---")
st.markdown("Created with ❤️ using Streamlit, PyTorch, MiDaS, rembg, and Trimesh.")
st.markdown("<p style='font-size:0.8em;'>Disclaimer: This is a demonstration project. Results depend heavily on image quality and complexity. Not for professional use without significant refinement.</p>", unsafe_allow_html=True)