import open3d as o3d
import numpy as np

# Create a sphere mesh
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
sphere.compute_vertex_normals()

# Create a visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add the sphere to the visualizer
vis.add_geometry(sphere)

# You can control the view here (e.g., set view angle, zoom, etc.)
view_control = vis.get_view_control()

# Set camera parameters for the view
front = np.array([1.0, 1.0, 1.0])  # Camera front direction
lookat = np.array([0.0, 0.0, 0.0])  # Camera look-at point (e.g., center of geometry)
up = np.array([0.0, 0.0, 1.0])  # Camera "up" direction
zoom = 0.8  # Camera zoom level (default is 1.0)

view_control.set_front(front)
view_control.set_lookat(lookat)
view_control.set_up(up)
view_control.set_zoom(zoom)

# Start the visualizer loop
vis.run()

# Destroy the window after closing
vis.destroy_window()
