import numpy as np
import pdb
# Function to create point cloud file
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


if __name__ == '__main__':
    # Define name for output file
    output_file = 'depth.ply'
    depth = np.load("depth.npy")
    depth = np.float32(depth)
    height = depth.shape[0]
    width = depth.shape[1]
    x,y = np.mgrid[0:height:1, 0:width:1]*0.05
    depth = np.expand_dims(depth, 2)
    x = np.expand_dims(x, 2)
    y = np.expand_dims(y, 2)
    points = np.concatenate((x, y, depth), axis=2)
    points = points.reshape(-1, 3)
    points_nonzero_index = np.argwhere(points[:,2] > 2) 
    points = points[points_nonzero_index,:]
    
#   43867是我的点云的数量，用的时候记得改成自己的
    one = np.ones(points.shape)
    one = np.float32(one)*255
#    points_3D = np.array([[1,2,3],[3,4,5]]) # 得到的3D点（x，y，z），即2个空间点
#    colors = np.array([[0, 255, 255], [0, 255, 255]])   #给每个点添加rgb
    # Generate point cloud
    print("\n Creating the output file... \n")
#    create_output(points_3D, colors, output_file)
    create_output(points, one, output_file)
