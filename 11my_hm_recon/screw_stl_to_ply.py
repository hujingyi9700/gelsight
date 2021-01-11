import numpy as np
from stl import mesh
import open3d as o3d
def visualize(mesh):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    vis.run()
    vis.destroy_window()
def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])
    np.savetxt(filename, vertices, fmt='%f %f %f %d %d %d')     # 必须先写入，然后利用write()在头部插入ply header
    ply_header = '''ply
    		format ascii 1.0
    		element vertex %(vert_num)d
    		property float x
    		property float y
    		property float z
    		property uchar red
    		property uchar green
    		property uchar blue
    		end_header
    		\n
    		'''
    with open(filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(vertices)))
        f.write(old)

def main():
    mesh = o3d.io.read_triangle_mesh("stl/sc5.STL")
    # pcd =  mesh.sample_points_poisson_disk(10000)
    pcd = mesh.sample_points_uniformly(number_of_points=500000)
    # print(np.asarray(pcd.points).shape)
    points = np.asarray(pcd.points)
    one = np.ones((500000 ,3))
    one = np.float32(one)*255
#    points_3D = np.array([[1,2,3],[3,4,5]]) # 得到的3D点（x，y，z），即2个空间点
#    colors = np.array([[0, 255, 255], [0, 255, 255]])   #给每个点添加rgb
    # Generate point cloud
    print("\n Creating the output file... \n")
#    create_output(points_3D, colors, output_file)
    output_file = 'screw.ply'
    create_output(points, one, output_file)
    o3d.visualization.draw_geometries([pcd])
    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    for alpha in np.logspace(np.log10(0.5), np.log10(0.01), num=4):
        print(f"alpha={alpha:.3f}")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha, tetra_mesh, pt_map)
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    alpha = 0.03
    print(f"alpha={alpha:.3f}")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    # pcd = mesh.sample_points_uniformly(number_of_points=5000)
    # visualize(pcd)

main()