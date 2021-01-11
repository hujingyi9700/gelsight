import open3d as o3d
import numpy as np
import copy
# from libs.math.compute_reverse import *
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(source, target, voxel_size):
    print(":: Load two point clouds and disturb initial pose.")


    #
    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    # source.transform(trans_init)
    # draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(40000000, 500))
    return result



def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
    distance_threshold = voxel_size * 1
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return result

def compute_trans(source, target, voxel_size):
    # voxel_size = 0.1  # means 5cm for this dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source=source, target=target,
                                                                                         voxel_size=voxel_size)
    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
                                     voxel_size,result_ransac)
    print(result_icp)
    print(result_icp.transformation)
    # print(compute_inverse_pose(result_icp.transformation))
    draw_registration_result(source_down, target_down, result_icp.transformation)

    # print(result_ransac)
    # print(result_icp.transformation)
    # print(compute_inverse_pose(result_icp.transformation))
    # draw_registration_result(source_down, target_down, result_ransac.transformation)

    return #result_icp.transformation, compute_inverse_pose(result_icp.transformation)
#  # Read the ply point cloud file in the computer
voxel_size = 0.5
source = o3d.io.read_point_cloud("depth.ply")  # source is the point cloud that needs to be registered
target = o3d.io.read_point_cloud("screw.ply")  # target is the target point cloud



compute_trans(source,target,voxel_size)

# print(result_ransac.transformation)
# draw_registration_result(source_down, target_down, result_ransac.transformation)

# def execute_fast_global_registration(source_down, target_down, source_fpfh,
#                                      target_fpfh, voxel_size):
#     distance_threshold = voxel_size * 0.5
#     print(":: Apply fast global registration with distance threshold %.3f" \
#             % distance_threshold)
#     result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
#         source_down, target_down, source_fpfh, target_fpfh,
#         o3d.pipelines.registration.FastGlobalRegistrationOption(
#             maximum_correspondence_distance=distance_threshold))
#     return result
#
# result_fast = execute_fast_global_registration(source_down, target_down,
#                                                source_fpfh, target_fpfh,
#                                                voxel_size)
# print(result_fast)
# draw_registration_result(source_down, target_down, result_fast.transformation)


