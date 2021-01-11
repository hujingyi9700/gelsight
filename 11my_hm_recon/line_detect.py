# -*- coding: UTF-8 -*-
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
from color_corre import color_corre

def matching(test_img, ref_blur,cali,table):
    diff = test_img - ref_blur
    
    diff[:,:,0] = np.clip((diff[:,:,0] - cali.blue_range[0])*cali.ratio, 0, cali.blue_bin-1)
    diff[:,:,1] = np.clip((diff[:,:,1] - cali.green_range[0])*cali.ratio, 0, cali.green_bin-1)
    diff[:,:,2] = np.clip((diff[:,:,2] - cali.red_range[0])*cali.ratio, 0, cali.green_bin-1)
    diff = diff.astype(int)
    grad_img = table[diff[:,:,0], diff[:,:,1],diff[:,:,2], :]
    return grad_img

def matching_v2(test_img, ref_blur,cali,table, blur_inverse):
    
    diff_temp1 = test_img - ref_blur
    diff_temp2 = diff_temp1 * blur_inverse
    diff_temp3 = np.clip((diff_temp2-cali.zeropoint)/cali.lookscale,0,0.999)
    diff = (diff_temp3*cali.bin_num).astype(int)
    # print(diff[150,160,:])
    # cv2.imshow('11',test_img.astype(np.uint8))
    
    # cv2.imshow('21',ref_blur.astype(np.uint8))
    # cv2.imshow('31',test_img)
    # cv2.imshow('41',test_img)
    # cv2.waitKey(0)

    plt.figure()
    plt.imshow(diff)
    plt.savefig("9.jpg")
    # pdb.set_trace()
    # plt.show()
    grad_img = table[diff[:,:,0], diff[:,:,1],diff[:,:,2], :] #從顏色對應到梯度了
    return grad_img
    
    
def show_depth(depth, figure_num):
    fig = plt.figure(figure_num)
    ax = fig.gca(projection='3d')
    # ax.set_aspect('equal')
    X = np.arange(0, depth.shape[1], 1)*0.05
    Y = np.arange(0, depth.shape[0], 1)*0.05
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, depth, cmap=cm.jet)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig("11")
#    plt.show()

def contact_detection(raw_image, ref_blur,marker_mask, kernel):
    diff_img = np.max(np.abs(raw_image.astype(np.float32) - ref_blur),axis = 2)
    # cv2.imshow("1",diff_img.astype(np.uint8))
    # cv2.waitKey(0)
    # pdb.set_trace()
    contact_mask = (diff_img> 15).astype(np.uint8)  #*(1-marker_mask)   #超參
    # cv2.imshow("1",(contact_mask*255).astype(np.uint8))
    # cv2.waitKey(0)
    # pdb.set_trace()
    contact_mask = cv2.dilate(contact_mask, kernel, iterations=1)  #超參
    contact_mask = cv2.erode(contact_mask, kernel, iterations=1)   
    return contact_mask
    
def marker_detection(raw_image_blur):
    m, n = raw_image_blur.shape[1], raw_image_blur.shape[0]
    raw_image_blur = cv2.pyrDown(raw_image_blur).astype(np.float32)
    ref_blur = cv2.GaussianBlur(raw_image_blur, (25, 25), 0)
    diff = ref_blur - raw_image_blur #圖像差分，以獲得邊界信息
    diff *= 16.0
    # print(diff) #float32和uint8不能亂轉換，因爲float32有正負，但uint8沒有
    # cv2.imshow('blur2', diff.astype(np.uint8))
    # cv2.waitKey(0)

    diff[diff < 0.] = 0.
    diff[diff > 255.] = 255.
    # print(diff)
    # diff = cv2.GaussianBlur(diff, (5, 5), 0)
    # cv2.imshow('diff', diff.astype(np.uint8))
    # cv2.waitKey(0)
    
    mask = (diff[:, :, 0] > 5) & (diff[:, :, 2] > 5) & (diff[:, :, 1] > 80) #超參,輸出爲1/0陣  #超參

    # cv2.imshow('mask', mask.astype(np.uint8) * 255)
    # cv2.waitKey(0)

    mask = cv2.resize(mask.astype(np.uint8), (m, n))
    
    # cv2.imshow('mask', mask.astype(np.uint8) * 255)
    # cv2.waitKey(0)

    return (1 - mask) * 255#mask 

def make_kernal(n,k_type):
    if k_type == 'circle':
        kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n)) #成形态学操作中用到的核
    else:
        kernal = cv2.getStructuringElement(cv2.MORPH_RECT,(n,n))
    return kernal 
#
    
if __name__ == '__main__': 
    table2 = np.load('table_0_smooth.npy')
    kernel1 = make_kernal(3,'circle')
    kernel2 = make_kernal(15,'circle')#25  #超參

#    plt.imshC:\Users\siyua\Documents\GitHub\gelsight_heightmap_reconstruction\matlab_version\calibration functionsow(table[:,:,80,0])
#    plt.show()
    imp = image_processor()
    cali = calibration()
    padx = 60  #超參
    pady = 10  #超參
    ref_img = cv2.imread('./test_data/ref.jpg')
#    ref_img = test_img.copy()
    # ref_img = color_corre(ref_img)
    ref_img = imp.undistort(ref_img)
    ref_img = imp.crop_image(ref_img, padx, pady) 
    marker = cali.mask_marker(ref_img)
    keypoints = cali.find_dots(marker) 
    marker_mask = cali.make_mask(ref_img, keypoints)
    # marker_image = np.dstack((marker_mask, marker_mask, marker_mask))
    ref_img = cv2.inpaint(ref_img,marker_mask,3,cv2.INPAINT_TELEA)
    ref_blur = cv2.GaussianBlur(ref_img.astype(np.float32), (3, 3), 0) + 1
#    ref_blur_small = cv2.pyrDown(ref_blur).astype(np.float32)
    blur_inverse = 1 + ((np.mean(ref_blur)/ref_blur)-1)*2
    
    
    
    test_img = cv2.imread('./test_data/l3.jpg')
    # test_img = color_corre(test_img)
    test_img = imp.undistort(test_img)
    test_img = imp.crop_image(test_img, padx, pady)
    # test_img = cv2.GaussianBlur(test_img.astype(np.float32), (3, 3), 0)
#    t1 = time.time()
    # marker_mask = marker_detection(test_img) #得到testimg的markermask  #超參
    marker = cali.mask_marker(test_img)
    keypoints = cali.find_dots(marker) 
    marker_mask = cali.make_mask(test_img, keypoints)
    # marker_image = np.dstack((marker_mask, marker_mask, marker_mask))
    test_img = cv2.inpaint(test_img,marker_mask,3,cv2.INPAINT_TELEA)
    gray = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)  
    


    # img = cv2.GaussianBlur(gray,(3,3),0) # 用高斯平滑处理原图像降噪。若效果不好可调节高斯核大小
    
    # canny = cv2.Canny(img, 10, 30)     # 调用Canny函数，指定最大和最小阈值，其中apertureSize默认为3。
    # cv2.imshow('mask', canny)
    # cv2.waitKey(0)
    # pdb.set_trace()



    def CannyThreshold(lowThreshold):
        detected_edges = cv2.GaussianBlur(gray,(3,3),0)
        detected_edges = cv2.Canny(detected_edges,
                                lowThreshold,
                                lowThreshold*ratio,
                                apertureSize = kernel_size)
        dst = cv2.bitwise_and(img,img,mask = detected_edges)  # just add some colours to edges from original image.
        cv2.imshow('canny demo',dst)

    lowThreshold = 0
    max_lowThreshold = 100
    ratio = 3
    kernel_size = 3
    
    img = test_img
    
    
    cv2.namedWindow('canny demo')
    
    cv2.createTrackbar('Min threshold','canny demo',lowThreshold, max_lowThreshold, CannyThreshold)
    
    CannyThreshold(0)  # initialization
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()