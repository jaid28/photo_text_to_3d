import trimesh
import numpy as np
from PIL import Image
import torch
from transformers import pipeline
import open3d as o3d  # Add open3d for mesh reconstruction

def image_to_3d(image):
    """
    Convert preprocessed image to 3D model using depth estimation and mesh reconstruction.
    Returns: Trimesh object.
    """
    # Load depth estimation model
    depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")

    # Convert image to PIL format
    image_pil = Image.fromarray(image)

    # Estimate depth
    depth = depth_estimator(image_pil)['predicted_depth'][0]
    depth = depth.numpy()

    # Create point cloud from depth map
    h, w = depth.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    z = depth  # Depth values
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Estimate normals (required for Poisson reconstruction)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Remove outliers to reduce noise (e.g., background artifacts)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Perform Poisson surface reconstruction
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

    # Convert Open3D mesh to Trimesh for export
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    return trimesh_mesh

def text_to_3d(prompt):
    """
    Generate 3D model from text prompt with more detailed predefined meshes.
    Returns: Trimesh object.
    """
    prompt = prompt.lower()

    if "car" in prompt:
        # Create a more detailed car: body + wheels
        body = trimesh.creation.box(extents=(2, 1, 0.5))  # Car body
        wheel1 = trimesh.creation.cylinder(radius=0.2, height=0.1)  # Wheel
        wheel1.apply_translation([-0.7, -0.4, -0.3])  # Front-left wheel
        wheel2 = trimesh.creation.cylinder(radius=0.2, height=0.1)
        wheel2.apply_translation([-0.7, 0.4, -0.3])  # Front-right wheel
        wheel3 = trimesh.creation.cylinder(radius=0.2, height=0.1)
        wheel3.apply_translation([0.7, -0.4, -0.3])  # Rear-left wheel
        wheel4 = trimesh.creation.cylinder(radius=0.2, height=0.1)
        wheel4.apply_translation([0.7, 0.4, -0.3])  # Rear-right wheel
        mesh = trimesh.util.concatenate([body, wheel1, wheel2, wheel3, wheel4])
    elif "chair" in prompt:
        # Create a more detailed chair: seat + backrest + legs
        seat = trimesh.creation.box(extents=(1, 1, 0.2))  # Seat
        backrest = trimesh.creation.box(extents=(1, 0.2, 1))  # Backrest
        backrest.apply_translation([0, 0, 0.6])  # Position above seat
        leg1 = trimesh.creation.cylinder(radius=0.05, height=0.5)  # Leg
        leg1.apply_translation([-0.4, -0.4, -0.35])  # Front-left leg
        leg2 = trimesh.creation.cylinder(radius=0.05, height=0.5)
        leg2.apply_translation([-0.4, 0.4, -0.35])  # Front-right leg
        leg3 = trimesh.creation.cylinder(radius=0.05, height=0.5)
        leg3.apply_translation([0.4, -0.4, -0.35])  # Rear-left leg
        leg4 = trimesh.creation.cylinder(radius=0.05, height=0.5)
        leg4.apply_translation([0.4, 0.4, -0.35])  # Rear-right leg
        mesh = trimesh.util.concatenate([seat, backrest, leg1, leg2, leg3, leg4])
    else:
        mesh = trimesh.creation.icosphere(radius=1)  # Default shape
    return mesh
