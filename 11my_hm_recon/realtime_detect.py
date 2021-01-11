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

    # plt.figure()
    # plt.imshow(diff)
    # plt.savefig("9.jpg")
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
    surf = ax.plot_surface(X, Y, depth, cmap=plt.get_cmap('rainbow'),vmin = 0, vmax = 5)
    # ax.contourf(X,Y,Z,zdir='z',offset=-2,cmap='rainbow')
    plt.pause(0.01)
    plt.cla()
    

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

    return mask 

def make_kernal(n,k_type):
    if k_type == 'circle':
        kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n)) #成形态学操作中用到的核
    else:
        kernal = cv2.getStructuringElement(cv2.MORPH_RECT,(n,n))
    return kernal 
#
    
if __name__ == '__main__': 
    #載入梯度-顏色查找表
    table2 = np.load('table_0_smooth.npy')
    
    #處理參考背景圖像
    kernel1 = make_kernal(3,'circle')
    kernel2 = make_kernal(15,'circle')#25  #超參

#    plt.imshC:\Users\siyua\Documents\GitHub\gelsight_heightmap_reconstruction\matlab_version\calibration functionsow(table[:,:,80,0])
#    plt.show()
    imp = image_processor()
    cali = calibration()
    padx = 60  #超參
    pady = 10  #超參
    ref_img = cv2.imread('./test_data/ref.jpg')
    # ref_img = color_corre(ref_img)
#    ref_img = test_img.copy()
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
    
    
    #讀取實時圖像並生成高度圖
    cam = cv2.VideoCapture(0)
    fig = plt.figure(dpi=120)
    ax = Axes3D(fig)
    plt.title('Point cloud')                                                                                                                                                                                    
    plt.ion()

    k = 0

    #相機開啓延時一下直到穩定
    while True:
        _, test_img = cam.read()
        # cv2.imshow('1', test_img)
        # cv2.waitKey(0)
        k+=1
        if k == 65:
            break

    while True:
        _, test_img = cam.read()
        test_img = color_corre(test_img)
        test_img = imp.undistort(test_img)
        test_img = imp.crop_image(test_img, padx, pady)
        cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)  
        cv2.resizeWindow('Frame', 640 ,480)
        cv2.moveWindow('Frame',1000,100)
        cv2.imshow('Frame', test_img)
        # cv2.imshow('1', test_img)
        # cv2.waitKey(0)
        # pdb.set_trace()
    # test_img = cv2.imread('./test_data/16.jpg')
        
        
        test_img = cv2.GaussianBlur(test_img.astype(np.float32), (3, 3), 0)
    #    t1 = time.time()
        marker_mask = marker_detection(test_img) #得到testimg的markermask  #超參
        
        marker_mask = cv2.dilate(marker_mask, kernel1, iterations=1) 

        # cv2.imshow("1",marker_mask*255)
        # key = cv2.waitKey(1)
        # if key == "q":
        #     break
        # pdb.set_trace()

        contact_mask = contact_detection(test_img, ref_blur, marker_mask, kernel2)
        # print(contact_mask)
        # cv2.imshow("1",contact_mask*255)
        # cv2.waitKey(0)
        # pdb.set_trace()
    
        # mask_2_show = np.dstack((np.zeros_like(marker_mask), marker_mask, np.zeros_like(marker_mask)))*40 +  test_img.astype(np.uint8)
        # plt.figure(20)
        # plt.imshow(mask_2_show)
        # plt.savefig("5.jpg")
        
        # plt.figure(21)
        # plt.imshow(contact_mask)
        # plt.savefig("6.jpg")
        
        # plt.show()
        
        grad_img2 = matching_v2(test_img, ref_blur, cali, table2, blur_inverse)
        
        grad_img2[:,:,0] = grad_img2[:,:,0] * (1-marker_mask)
        grad_img2[:,:,1] = grad_img2[:,:,1] * (1-marker_mask)

    #    depthgx = np.array(gx.ImGradX)1 = fast_poisson(grad_img1[:,:,0], grad_img1[:,:,1])
        depth2 = fast_poisson(grad_img2[:,:,0], grad_img2[:,:,1])
        
    # depth1[depth1<0] = 0
        depth2[depth2<0] = 0
        show_depth3d(depth2)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()