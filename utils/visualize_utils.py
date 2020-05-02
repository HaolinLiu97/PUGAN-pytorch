import open3d as o3d

def visualize_point_cloud(xyz):
    '''
    Args:
        xyz is of shape N,3
    '''
    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(xyz)
    vis=o3d.visualization.Visualizer()

    vis.create_window()
    vis.add_geometry(pcd)
    img=vis.capture_screen_float_buffer(True)

    return img



