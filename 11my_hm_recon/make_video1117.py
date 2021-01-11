import numpy as np 
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from fast_poisson import fast_poisson
import cv2
import matplotlib.pyplot as plt
#from fast_poisson import poisson_reconstruct.
from calibration import image_processor, calibration
import time
import pdb
import os


path = '/home/hjy/project/vscodePr/python/7my_hm_recon/'

def show_depth3d(depth):
    X = np.arange(0, depth.shape[1], 1)*0.05
    Y = np.arange(0, depth.shape[0], 1)*0.05
    X, Y = np.meshgrid(X, Y)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_xlim3d(xmin=0, xmax=30)
    # ax.set_ylim3d(ymin=0, ymax=30)
    ax.set_zlim3d(zmin=0, zmax=10)

    plt.ioff()
    surf = ax.plot_surface(X, Y, depth, cmap=plt.get_cmap('rainbow'))
    cloud_img = plt.gca()
    # ax.contourf(X,Y,Z,zdir='z',offset=-2,cmap='rainbow')
    # plt.pause(0.01)
    # plt.cla()
    return cloud_img

def make_video(frame_list, name):
    (w, h, _) = frame_list[0].shape
    video = cv2.VideoWriter(path+ 'video/' + name + '.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5, (h, w)) #這裏是反的

    for item in frame_list:
        video.write(item)

    video.release()
    cv2.destroyAllWindows()
    print('视频合成生成完成啦')
    
if __name__ == '__main__': 
    
    #原圖
    raw_img_list = np.load(path + 'video/raw_img.npy')
    make_video(raw_img_list, 'raw_video')
    
    #點雲
    cloud_img_list = []

    fig = plt.figure(dpi=120)
    ax = Axes3D(fig)
    plt.title('Point cloud')                                                                                                                                                                                    
    plt.ion()


    depth_list = np.load(path + 'video/depth.npy')
    for i in range(len(depth_list)):
        cloud_img = show_depth3d(depth_list[i])
        plt.savefig(path + 'cloud_img/'+str(i)+'.jpg')
    # pdb.set_trace()

    file_name = os.listdir(path+'cloud_img/')
    for f in file_name:
        img_path = path +'cloud_img/'+f
        cloud_img = cv2.imread(img_path)
        cloud_img_list.append(cloud_img)
    
    make_video(cloud_img_list, 'cloud_video')