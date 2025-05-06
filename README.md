# Photo/Text to 3D Model Prototype

This prototype converts a photo (single object) or text prompt into a simple 3D model (.obj/.stl) with visualization, as per the assignment requirements.

## Project Overview

This project demonstrates a pipeline to:
- Convert a photo of a single object (e.g., chair, car, toy) into a 3D model using image preprocessing and depth estimation.
- Generate a 3D model from a text prompt using a simplified mapping approach.
- Output the model as a `.obj` or `.stl` file and visualize it using `pyrender`.

## Setup

1. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
Install Dependencies:
pip install -r requirements.txt

Note: If torch installation is slow or fails, use the CPU version:
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu

2. **Prepare Input**:
The input/ folder contains sample inputs:
sample.jpg: A photo of a chair (provided as a sample).
prompt.txt: A text prompt ("A small toy car").
Add your own .jpg/.png images or .txt prompts to the input/ folder for testing.

## Usage
Run the main script with either an image or text input:

For Image Input:
  '''bash
  python main.py --input input/sample.jpg --output output/model.obj

For Text Input:
  '''bash
  python main.py --input input/prompt.txt --output output/model.stl

## Output:
A 3D model is saved to the output/ folder (e.g., output/model.obj or output/model.stl).
The model is visualized in a 3D viewer using pyrender (or matplotlib if pyrender fails).

## Libraries Used
a. rembg: Background removal for images.
b. opencv-python: Image preprocessing.
c. trimesh: 3D mesh creation and export to .obj/.stl.
d. pyrender: 3D model visualization.
e. transformers, torch: Depth estimation using the MiDaS model (Intel/dpt-large).
f. numpy, Pillow: Image and array handling.

3. **Optional**: Lightweight Shap-E Integration (If You Have a GPU):
If you have access to a GPU and can install Shap-E, you can integrate it for true text-to-3D generation. Note: This requires significant setup, so it’s optional. Here’s a basic outline:

Install Shap-E:
pip install git+https://github.com/openai/shap-e.git

## Limitations
Image-to-3D:
Single-view depth estimation may still miss fine details (e.g., thin chair legs), but Poisson reconstruction with open3d improves completeness and reduces noise.
rembg may struggle with semi-transparent backgrounds, but an OpenCV-based fallback improves object isolation, reducing artifacts.

Text-to-3D:
Still uses predefined meshes as the primary approach, but they are now more detailed (e.g., cars with wheels). Full text-to-3D models like Shap-E are optional due to computational requirements.

Visualization:
pyrender requires OpenGL support, but a matplotlib fallback ensures visualization works on most systems, though with less detail.

## Future Improvements
a. Integrate advanced text-to-3D models (e.g., Point-E, Shap-E) for more accurate generation from text prompts.
b. Use multi-view reconstruction (e.g., COLMAP) to improve 3D model accuracy from images.
c. Enhance visualization with a web-based viewer (e.g., three.js) for broader compatibility.
d. Add mesh smoothing and texture mapping to improve the visual quality of the 3D models.

## Troubleshooting
Dependency Installation:
If rembg fails, ensure libgl1-mesa-glx is installed (sudo apt-get install libgl1-mesa-glx on Ubuntu).
If pyrender fails, try pip install pyrender --no-deps and install dependencies manually, or ensure OpenGL is configured.
If open3d fails, ensure you have a compatible Python version (e.g., 3.8–3.10) and try pip install open3d --no-cache-dir.
Use CPU-based torch if GPU installation fails (see Setup section).

Background Removal:
If rembg fails (e.g., with complex backgrounds), the pipeline falls back to OpenCV-based segmentation. For best results, use images with high-contrast backgrounds or manually crop the image.

Visualization:
If pyrender fails to open, the pipeline falls back to matplotlib. Alternatively, load the .obj/.stl file in Blender, MeshLab, or an online viewer like https://3dviewer.net.


