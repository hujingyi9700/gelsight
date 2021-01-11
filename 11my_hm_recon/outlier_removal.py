import open3d as o3d

#读取电脑中的 ply 点云文件
source = o3d.read_point_cloud("cloud/plys/6.ply")  #source 为需要配准的点云
target = o3d.read_point_cloud("cloud/plys/0.ply")  #target 为目标点云

#为两个点云上上不同的颜色
source.paint_uniform_color([1, 0.706, 0])    #source 为黄色
target.paint_uniform_color([0, 0.651, 0.929])#target 为蓝色

#为两个点云分别进行outlier removal
processed_source, outlier_index = o3d.geometry.radius_outlier_removal(source,
                                              nb_points=16,
                                              radius=0.5)

processed_target, outlier_index = o3d.geometry.radius_outlier_removal(target,
                                              nb_points=16,
                                              radius=0.5)

#o3d.geometry.radius_outlier_removal 这个函数是使用球体判断一个特例的函数，它需要
#两个参数：nb_points 和 radius。 它会给点云中的每个点画一个半径为 radius 的球体，如
#果在这个球体中其他的点的数量小于 nb_points, 这个算法会将这个点判断为特例，并删除。


#显示两个点云
#o3d.visuzlization_draw_geometries 是一个更加简单的显示点云的函数，它不需要创建
#一个visualizer类，直接调用这个函数，在参数里放一个包含你想显示的点云的list就行了。
o3d.visualization.draw_geometries([processed_source,processed_target])

