# Photo or Text to Simple 3D Model Prototype

This project is a Streamlit application that generates a basic 3D model (.obj or .stl) from either a single photo of an object or a short text prompt.

## Features

-   **Input:**
    -   Image (.jpg/.png) of a single object (e.g., chair, car, toy).
    -   Short text prompt (e.g., "A small toy car").
-   **Processing:**
    -   **For Photo Input:**
        -   Background removal using `rembg`.
        -   Depth estimation using `MiDaS`.
        -   Pseudo-3D model generation by converting the depth map into a heightmap mesh.
    -   **For Text Input:**
        -   3D model generation using OpenAI's `Shap-E` model.
-   **Output:**
    -   Downloadable .obj and .stl files.
    -   In-app 3D visualization of the generated model using `PyVista` (via `stpyvista`) or `Matplotlib`.

## Setup and Installation

1.  **Clone the repository (if applicable) or create project files.**

2.  **Create a Python virtual environment:**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    -   Windows (PowerShell/CMD):
        ```bash
        .\venv\Scripts\activate
        ```
    -   macOS/Linux (bash/zsh):
        ```bash
        source venv/bin/activate
        ```

4.  **Install dependencies:**
    Ensure you have `pip` upgraded:
    ```bash
    pip install --upgrade pip
    ```
    Install from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
    **Note on PyTorch/CUDA:**
    The `requirements.txt` includes `torch`, `torchvision`, and `torchaudio`. `pip` will attempt to install a version compatible with your system. If you have an NVIDIA GPU and want to leverage CUDA, ensure your CUDA toolkit and drivers are installed. You might need to install a specific PyTorch build. Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) for custom commands.
    For `rembg`, `rembg[gpu]` is specified. If you don't have a compatible GPU or encounter issues, change it to `rembg[cpu]` in `requirements.txt` and reinstall.

    **Note on Shap-E:**
    The first time you run the text-to-3D generation, Shap-E models (which can be large) will be downloaded. This requires an internet connection and might take some time.

5.  **Run the Streamlit Application:**
    ```bash
    streamlit run app.py
    ```
    The application will open in your web browser.

## Libraries Used

-   **`streamlit`**: For creating the web application interface.
-   **`torch`, `torchvision`**: PyTorch library, used by MiDaS and Shap-E.
-   **`transformers`**: Hugging Face library (potentially used by Shap-E or other future models).
-   **`rembg`**: For background removal from images.
-   **`trimesh`**: For 3D mesh processing (creation, manipulation, import/export of OBJ/STL).
-   **`scipy`**: For scientific computing, specifically `Delaunay` triangulation for mesh generation from depth maps.
-   **`Pillow` (PIL)**: For image manipulation.
-   **`numpy`**: For numerical operations.
-   **`pyvista`**: For 3D plotting and mesh visualization.
-   **`stpyvista`**: Streamlit component to embed PyVista plots.
-   **`shap-e`**: OpenAI's library for generating 3D assets from text or images.
-   **`MiDaS` (via `torch.hub`)**: For monocular depth estimation from images.

## Thought Process

1.  **Goal**: Create a simple prototype for photo/text to 3D model. "Simple" implies not aiming for state-of-the-art reconstruction but a functional pipeline.
2.  **UI**: `Streamlit` is ideal for rapid prototyping of data/ML applications.
3.  **Input Handling**: Radio buttons to select image or text input. File uploader for images, text input for prompts.
4.  **Image-to-3D Strategy**:
    *   True single-image 3D reconstruction is very hard.
    *   A "2.5D" approach using a depth map is more feasible for a prototype.
    *   **Preprocessing**: `rembg` for background removal to isolate the object. The alpha channel is key.
    *   **Depth Estimation**: `MiDaS` (from Intel) is a well-regarded monocular depth estimation model available via `torch.hub`.
    *   **Mesh Creation**:
        *   Treat the (inverted and masked) depth map as a height map.
        *   Create a 2D grid of points corresponding to object pixels.
        *   Use `scipy.spatial.Delaunay` to triangulate these points in the XY plane.
        *   Assign Z values from the height map.
        *   Add a simple extrusion/base to give it some volume and attempt to close the mesh.
        *   `trimesh` is used for all mesh object handling and export.
5.  **Text-to-3D Strategy**:
    *   Leverage an existing open-source model. OpenAI's `Shap-E` is a strong candidate that can generate meshes directly or via latents.
    *   The `shap-e` Python library provides convenient functions for this.
    *   Note that this can be computationally intensive and slow on CPU.
6.  **Output & Visualization**:
    *   Generated models are `trimesh.Trimesh` objects.
    *   Export to `.obj` and `.stl` using `trimesh.export()`.
    *   Visualize using `stpyvista` for interactive 3D plots within Streamlit. Fallback to `matplotlib` if `pyvista` is unavailable or problematic, though `matplotlib`'s 3D mesh rendering is basic.
7.  **Modularity & Caching**:
    *   Heavy models (`MiDaS`, `Shap-E`) are loaded using `st.cache_resource` to avoid reloading on every interaction.
    *   Helper functions for saving meshes and creating meshes from depth.
8.  **Dependencies & Environment**: Standard Python `venv` and `requirements.txt` for reproducibility.

## Limitations

-   **Image-to-3D Quality**: The "3D" model from an image is essentially a heightmap extrusion based on estimated depth. It won't capture the full 3D geometry of an object (e.g., the back or hollow parts). It works best for objects with clear frontal depth variation.
-   **Text-to-3D Quality & Speed**: Shap-E results vary based on the prompt and settings. Generation can be slow, especially without a GPU.
-   **Watertight Meshes**: Generated meshes, especially from the image-to-3D pipeline, may not always be perfectly watertight, which can be an issue for 3D printing. Some basic hole filling is attempted.
-   **Resource Requirements**: Both pipelines involve deep learning models that can be memory and computationally intensive. Shap-E, in particular, benefits greatly from a CUDA-enabled GPU.
