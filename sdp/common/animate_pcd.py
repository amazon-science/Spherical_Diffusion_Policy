import open3d as o3d
import numpy as np
import time

# Assuming 'action_pred' is a list of numpy arrays, each representing a point cloud
# Example: action_pred = [np.random.rand(100, 3) for _ in range(10)]  # Replace with your actual point clouds

def animate_point_clouds(action_pred, static_pcd=None, t=0.1, view=[1, 1, 1], up=[0, 0, 1], zoom=1, k=None):
    # Create an Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Set the view angle
    view_control = vis.get_view_control()

    # Set the camera parameters
    front = np.array(view)  # Camera front vector (direction the camera is looking at)
    lookat = np.array([0.0, 0.0, 0.0])  # Camera look-at point (where the camera is pointing)
    up = np.array(up)  # Camera up vector (which way is 'up' for the camera)
    zoom = zoom  # Zoom factor (1.0 is default, <1 zooms in, >1 zooms out)

    view_control.set_front(front)
    view_control.set_lookat(lookat)
    view_control.set_up(up)
    view_control.set_zoom(zoom)

    # Explicitly update the view control
    vis.update_geometry(static_pcd)
    vis.update_renderer()

    # Initialize an empty point cloud object for the visualizer
    pcd = o3d.geometry.PointCloud()

    if k is not None and k > -1:
        for i in range(200):
            # for points in action_pred:
            points = action_pred[k % len(action_pred)]
            # Convert the numpy array to Open3D PointCloud
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(points[:, 3:])

            # Clear previous geometry and add the new point cloud
            vis.clear_geometries()
            vis.add_geometry(pcd)
            vis.add_geometry(static_pcd)

            view_control.set_front(front)
            view_control.set_lookat(lookat)
            view_control.set_up(up)
            view_control.set_zoom(zoom)

            # Update the visualizer
            vis.poll_events()
            vis.update_renderer()

    elif k is None:
        for i in range(200):
            # for points in action_pred:
            points = action_pred[i % len(action_pred)]
            # Convert the numpy array to Open3D PointCloud
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(points[:, 3:])

            # Clear previous geometry and add the new point cloud
            vis.clear_geometries()
            vis.add_geometry(pcd)
            vis.add_geometry(static_pcd)

            view_control.set_front(front)
            view_control.set_lookat(lookat)
            view_control.set_up(up)
            view_control.set_zoom(zoom)

            # Update the visualizer
            vis.poll_events()
            vis.update_renderer()

            # Optional: Delay to slow down the animation (e.g., 100 ms per frame)
            time.sleep(t)
    else:
        for i in range(200):
            # Clear previous geometry and add the new point cloud
            vis.clear_geometries()
            vis.add_geometry(static_pcd)

            view_control.set_front(front)
            view_control.set_lookat(lookat)
            view_control.set_up(up)
            view_control.set_zoom(zoom)

            # Update the visualizer
            vis.poll_events()
            vis.update_renderer()


    # vis.destroy_window()


if __name__ == "__main__":
    # Example usage

    action_pred = [np.random.rand(100, 6) for _ in range(10)]  # Replace this with your actual list of point clouds
    static_pcd = o3d.geometry.PointCloud()
    static_pcd.points = o3d.utility.Vector3dVector(action_pred[0][:, :3])
    static_pcd.colors = o3d.utility.Vector3dVector(action_pred[0][:, 3:] * 0)
    animate_point_clouds(action_pred, static_pcd)
