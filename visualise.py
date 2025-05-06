import trimesh
import pyrender
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_3d_model(model_path):
    """
    Visualize 3D model using pyrender, with a fallback to matplotlib if pyrender fails.
    """
    # Load mesh
    mesh = trimesh.load(model_path)

    # Try pyrender first
    try:
        scene = pyrender.Scene()
        mesh_pyrender = pyrender.Mesh.from_trimesh(mesh)
        scene.add(mesh_pyrender)
        viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)
        print("Visualization successful with pyrender.")
    except Exception as e:
        print(f"pyrender failed: {e}. Falling back to matplotlib visualization.")

        # Fallback: Use matplotlib to plot the mesh
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Extract vertices and faces
        vertices = mesh.vertices
        faces = mesh.faces

        # Plot vertices
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=1)

        # Plot wireframe
        for face in faces:
            x = vertices[face, 0]
            y = vertices[face, 1]
            z = vertices[face, 2]
            ax.plot(np.append(x, x[0]), np.append(y, y[0]), np.append(z, z[0]), 'b-')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title("3D Model Visualization (Matplotlib Fallback)")
        plt.show()
