import cv2
import numpy as np 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt  
import glob
from fast_poisson import fast_poisson
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from skimage.restoration import inpaint
import pdb
from color_corre import color_corre
np.set_printoptions(threshold=np.inf)

class image_processor:
    def __init__(self):
        pass
    
    def crop_image(self,img, padx, pady):
        return img[pady:-pady,padx:-padx-50]
    def undistort(self,frame):
        fx = 237.7077
        cx = 335.2097
        fy = 241.1352
        cy = 228.8347
        k1, k2, p1, p2, k3 = -0.2368, 0.0511, 0.0, 0.0, 0.0
    
        # 相机坐标系到像素坐标系的转换矩阵
        k = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        # 畸变系数
        d = np.array([
            k1, k2, p1, p2, k3
        ])
        h, w = frame.shape[:2]
        mapx, mapy = cv2.initUndistortRectifyMap(k, d, None, k, (w, h), 5)
        return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)



class calibration:
    def __init__(self):
        self.BallRad= 4/2 #4.76/2 #mm
        self.Pixmm =  0.05#4.76/100 #0.0806 * 1.5 mm/pixel1.5/18.87 #0.05效果好
        self.ratio = 1/2.
        self.red_range = [-90, 90]
        self.green_range = [-90, 90] #[-60, 50]
        self.blue_range = [-90, 90] # [-80, 60]
        self.red_bin = int((self.red_range[1] - self.red_range[0])*self.ratio)
        self.green_bin = int((self.green_range[1] - self.green_range[0])*self.ratio)
        self.blue_bin = int((self.blue_range[1] - self.blue_range[0])*self.ratio)
        self.zeropoint = -90
        self.lookscale = 180
        self.bin_num = 90
    
    def mask_marker(self, raw_image):
    # 爲了生成marker的mask圖像，即生成在marker處是0，其餘爲255（0）的二值圖像,但是該圖像未給marker擬合成圓形，比較粗糙。
        m, n = raw_image.shape[1], raw_image.shape[0]
        raw_image = cv2.pyrDown(raw_image).astype(np.float32)
        blur = cv2.GaussianBlur(raw_image, (25, 25), 0)
        blur2 = cv2.GaussianBlur(raw_image, (5, 5), 0)
        diff = blur - blur2
        diff *= 16.0

        # cv2.imshow('blur2', blur2.astype(np.uint8))
        # cv2.waitKey(0)
        # pdb.set_trace()

        diff[diff < 0.] = 0.
        diff[diff > 255.] = 255.

        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        # cv2.imshow('diff', diff.astype(np.uint8))
        # cv2.waitKey(0)
        # pdb.set_trace()

        mask = (diff[:, :, 0] > 5) & (diff[:, :, 2] > 5) & (diff[:, :, 1] >80) # BGR 超參
        # cv2.imshow('mask', mask.astype(np.uint8) * 255)
        # cv2.waitKey(0)
        # pdb.set_trace()

        mask = cv2.resize(mask.astype(np.uint8), (m, n))
#        mask = mask * self.dmask
#        mask = cv2.dilate(mask, self.kernal4, iterations=1)

        # mask = cv2.erode(mask, self.kernal4, iterations=1)
        # a = (1 - mask) * 255
        # 
        # print(mask)
        return (1 - mask) * 255 #反色
    
    def find_dots(self, binary_image):
        #返回maker的中心點坐標
        # down_image = cv2.resize(binary_image, None, fx=2, fy=2)
        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        params.minThreshold = 1
        params.maxThreshold = 12
        params.minDistBetweenBlobs = 9
        params.filterByArea = True
        params.minArea = 9
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        params.minInertiaRatio = 0.5

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(binary_image.astype(np.uint8))
        
        # im_to_show = (np.stack((binary_image,)*3, axis=-1)-100)
        # for i in range(len(keypoints)):
        #     cv2.circle(im_to_show, (int(keypoints[i].pt[0]), int(keypoints[i].pt[1])), 5, (0, 100, 100), -1)
            
        # cv2.imshow('final_image1', im_to_show)
        # cv2.waitKey(100000)

        return keypoints
    
    def make_mask(self, img, keypoints):
        # 爲了生成marker的mask圖像，即生成在marker處是0，其餘爲255（0）的二值圖像,給marker擬合成了圓形。
        img = np.zeros_like(img[:,:,0])
        
        for i in range(len(keypoints)):
            # cv2.circle(img, (int(keypoints[i].pt[0]), int(keypoints[i].pt[1])), 6, (1), -1)
            cv2.ellipse(img, (int(keypoints[i].pt[0]), int(keypoints[i].pt[1])), (9, 9) ,0 ,0 ,360, (255), -1) #擬合的橢圓是超參！
            # print(int(keypoints[i].pt[0]), int(keypoints[i].pt[1]))
            # cv2.imshow('final_image1', img)
            # cv2.waitKey(0)

        return img
    
    def contact_detection(self,raw_image, ref, marker_mask):
        blur = cv2.GaussianBlur(ref.astype(np.float32), (3, 3), 0)
        diff_img = np.max(np.abs(raw_image.astype(np.float32) - blur),axis = 2)
        contact_mask = (diff_img> 13).astype(np.uint8)*(1-marker_mask)  #超參13
        
        # print(np.max(np.abs(raw_image.astype(np.float32) - blur),axis = 2))
        
        # cv2.imshow('mask', contact_mask*255)
        # cv2.imshow('mas', diff_img.astype(np.uint8))
        # cv2.waitKey(0)
        # pdb.set_trace()
       
        contours,_ = cv2.findContours(contact_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]
        sorted_areas = np.sort(areas)
        cnt=contours[areas.index(sorted_areas[-1])] #the biggest contour
        # print(cnt)
        # pdb.set_trace()
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        center = (int(x),int(y))
        radius = int(radius) #半徑
        
        key = -1
        while key != 27:
            center = (int(x),int(y))
            radius = int(radius)
            im2show = cv2.circle(np.array(raw_image),center,radius,(0,40,0),2)
            
            cv2.imshow('contact', im2show.astype(np.uint8))

            key = cv2.waitKey(0)
            if key == 119:
                y -= 1
            elif key == 115:
                y += 1
            elif key == 97:
                x -= 1
            elif key == 100:
                x += 1
            elif key == 109:
                radius += 1
            elif key == 110:
                radius -= 1

        contact_mask = np.zeros_like(contact_mask)
        cv2.circle(contact_mask,center,radius,(1),-1)

        contact_mask = contact_mask * (1-marker_mask) #1-(0or255)變成了0的地方是1，255的地方是2的矩陣
        
        cv2.imshow('contact_mask',contact_mask*255)
        cv2.waitKey(0)

        return contact_mask, center, radius

    def get_gradient(self, img, ref, center, radius_p, valid_mask, table, table_account):
        ball_radius_p = self.BallRad / self.Pixmm
        blur = cv2.GaussianBlur(ref.astype(np.float32), (3, 3), 0)
        img_smooth = cv2.GaussianBlur(img.astype(np.float32), (3, 3), 0)
        diff = img_smooth - blur 
#        diff_valid = np.abs(diff * np.dstack((valid_mask,valid_mask,valid_mask)))
        pixels_valid = diff[valid_mask>0]
        pixels_valid[:,0] = np.clip((pixels_valid[:,0] - self.blue_range[0])*self.ratio, 0, self.blue_bin-1)
        pixels_valid[:,1] = np.clip((pixels_valid[:,1] - self.green_range[0])*self.ratio, 0, self.green_bin-1)
        pixels_valid[:,2] = np.clip((pixels_valid[:,2] - self.red_range[0])*self.ratio, 0, self.red_bin-1)
        pixels_valid = pixels_valid.astype(int)
        
        
#        range_blue = [max(np.mean(pixels_valid[:,0])-2*np.std(pixels_valid[:,0]),np.min(pixels_valid[:,0])), \
#                     min(np.mean(pixels_valid[:,0])+2.5*np.std(pixels_valid[:,0]), np.max(pixels_valid[:,0]))] 
#        range_green = [max(np.mean(pixels_valid[:,1])-2*np.std(pixels_valid[:,1]),np.min(pixels_valid[:,1])), \
#                     min(np.mean(pixels_valid[:,1])+2.5*np.std(pixels_valid[:,1]), np.max(pixels_valid[:,1]))] 
#        range_red = [max(np.mean(pixels_valid[:,2])-2*np.std(pixels_valid[:,2]),np.min(pixels_valid[:,2])), \
#                     min(np.mean(pixels_valid[:,2])+2.5*np.std(pixels_valid[:,2]), np.max(pixels_valid[:,2]))] 
        
#        print('blue', range_blue, 'green', range_green, 'red',  range_red)
        
#        print(np.min(pixels_valid[:,0]), np.max(pixels_valid[:,0]))
#        print(np.min(pixels_valid[:,1]), np.max(pixels_valid[:,1]))
#        print(np.min(pixels_valid[:,2]), np.max(pixels_valid[:,2]))
#        print(pixels_valid.shape)
#        plt.figure(0)
#        plt.hist(pixels_valid[:,0], bins = 256)
#        plt.figure(1)
#        plt.hist(pixels_valid[:,1], bins = 256)
#        plt.figure(2)
#        plt.hist(pixels_valid[:,2], bins = 256)
#        plt.show()
        x = np.linspace(0, img.shape[0]-1,img.shape[0])
        y = np.linspace(0, img.shape[1]-1,img.shape[1])
        xv, yv = np.meshgrid(y, x)
#        print('img shape', img.shape, xv.shape, yv.shape)
        xv = xv - center[0]
        yv = yv - center[1]
        rv = np.sqrt(xv**2 + yv**2)

        radius_p = min(radius_p, ball_radius_p-1) 
        mask = (rv < radius_p)
        mask_small = (rv < radius_p-1)
#        gradmag=np.arcsin(rv*mask/ball_radius_p)*mask;
#        graddir=np.arctan2(-yv*mask, -xv*mask)*mask;
#        gradx_img=gradmag*np.cos(graddir);
#        grady_img=gradmag*np.sin(graddir);
#        depth = fast_poisson(gradx_img, grady_img)
        temp = ((xv*mask)**2 + (yv*mask)**2)*self.Pixmm**2
        height_map = (np.sqrt(self.BallRad**2-temp)*mask - np.sqrt(self.BallRad**2-(radius_p*self.Pixmm)**2))*mask
        height_map[np.isnan(height_map)] = 0
#        depth = poisson_reconstruct(grady_img, gradx_img, np.zeros(grady_img.shape))
        gx_num = signal.convolve2d(height_map, np.array([[0,0,0],[0.5,0,-0.5],[0,0,0]]), boundary='symm', mode='same')*mask_small
        gy_num = signal.convolve2d(height_map, np.array([[0,0,0],[0.5,0,-0.5],[0,0,0]]).T, boundary='symm', mode='same')*mask_small
#        depth_num = fast_poisson(gx_num, gy_num)
#        plt.imshow(height_map)
#        plt.show()
        gradxseq = gx_num[valid_mask>0]
        gradyseq = gy_num[valid_mask>0]
        
        for i in range(gradxseq.shape[0]):
            b, g, r = pixels_valid[i,0], pixels_valid[i,1], pixels_valid[i,2]
#            print(r,g,b)
            if table_account[b,g,r] < 1.: 
                table[b,g,r,0] = gradxseq[i]
                table[b,g,r,1] = gradyseq[i]
                table_account[b,g,r] += 1
            else:
#                print(table[b,g,r,0], gradxseq[i], table[b,g,r,1], gradyseq[i])
                table[b,g,r,0] = (table[b,g,r,0]*table_account[b,g,r] + gradxseq[i])/(table_account[b,g,r]+1)
                table[b,g,r,1] = (table[b,g,r,1]*table_account[b,g,r] + gradyseq[i])/(table_account[b,g,r]+1)
                table_account[b,g,r] += 1
        return table, table_account
        
      
    def get_gradient_v2(self, img, ref, center, radius_p, valid_mask, table, table_account):
        ball_radius_p = self.BallRad / self.Pixmm #把實際的半徑轉換成像素值的半徑，但是這個pixmm是如何獲得？ #超參
        blur = cv2.GaussianBlur(ref.astype(np.float32), (3, 3), 0) + 1 # 背景進行高斯模糊
        
        blur_inverse = 1+ ((np.mean(blur)/blur)-1)*2 #我看到的是背景rgb的三塊互補色

        # cv2.imshow("1",(blur_inverse*100).astype(np.uint8)) 
        # cv2.waitKey(0)
        # pdb.set_trace()


        img_smooth = cv2.GaussianBlur(img.astype(np.float32), (3, 3), 0)
        diff_temp1 = img_smooth - blur #我看到的是接觸區域出現了非常鮮明的綠紅藍三塊,主要是因爲減去了ref背景，float轉到uint8會把-1變成255，所以顯示出來是強烈的彩色
        diff_temp2 = diff_temp1 * blur_inverse #我看到2和1沒有什麼區別，數值上有區別
        # print(diff_temp1)
        # cv2.imshow("0",img_smooth.astype(np.uint8))
        # showimg = np.where(diff_temp2 > 0, diff_temp1, 0)
        # cv2.imshow("3",blur.astype(np.uint8))
        # # print(showimg)
        # cv2.imshow("1",(showimg+50).astype(np.uint8)) 
        # cv2.waitKey(0) 
        # pdb.set_trace()
        # # print(diff_temp1)
        # print(diff_temp1)#.astype(np.uint8))
        # print(img_smooth[100,100,2])
        # print(blur[100,100,2])
        
        # print(np.mean(diff_temp2), np.std(diff_temp2))
        # print(np.min(diff_temp2), np.max(diff_temp2))
        diff_temp3 = np.clip((diff_temp2-self.zeropoint)/self.lookscale,0,0.999) #將temp2+90除180，将数组中的元素限制在0, 0.999之间，我看到的是白底黑點，明顯的接觸區域的紅綠藍
        
        
        diff = (diff_temp3*self.bin_num).astype(int) #我看到的是白底，接觸區域只有藍色的塊
        # cv2.imshow("1",diff.astype(np.uint8)) 
        # cv2.waitKey(0)
        # pdb.set_trace()
        
#        diff_valid = np.abs(diff * np.dstack((valid_mask,valid_mask,valid_mask)))
        pixels_valid = diff[valid_mask>0] #只保留validmask中不爲0位置的值，形成另外一個更小的低維矩陣
        # print(pixels_valid)
        # pdb.set_trace()
        
        # pixels_valid[:,0] = np.clip((pixels_valid[:,0] - self.blue_range[0])*self.ratio, 0, self.blue_bin-1)
        # pixels_valid[:,1] = np.clip((pixels_valid[:,1] - self.green_range[0])*self.ratio, 0, self.green_bin-1)
        # pixels_valid[:,2] = np.clip((pixels_valid[:,2] - self.red_range[0])*self.ratio, 0, self.red_bin-1)
        # pixels_valid = pixels_valid.astype(int)
            
            
        # range_blue = [max(np.mean(pixels_valid[:,0])-2*np.std(pixels_valid[:,0]),np.min(pixels_valid[:,0])), \
        #                 min(np.mean(pixels_valid[:,0])+2.5*np.std(pixels_valid[:,0]), np.max(pixels_valid[:,0]))] 
        # range_green = [max(np.mean(pixels_valid[:,1])-2*np.std(pixels_valid[:,1]),np.min(pixels_valid[:,1])), \
        #                 min(np.mean(pixels_valid[:,1])+2.5*np.std(pixels_valid[:,1]), np.max(pixels_valid[:,1]))] 
        # range_red = [max(np.mean(pixels_valid[:,2])-2*np.std(pixels_valid[:,2]),np.min(pixels_valid[:,2])), \
        #                 min(np.mean(pixels_valid[:,2])+2.5*np.std(pixels_valid[:,2]), np.max(pixels_valid[:,2]))] 
            
        # print('blue', range_blue, 'green', range_green, 'red',  range_red)
            
        # print(np.min(pixels_valid[:,0]), np.max(pixels_valid[:,0]))
        # print(np.min(pixels_valid[:,1]), np.max(pixels_valid[:,1]))
        # print(np.min(pixels_valid[:,2]), np.max(pixels_valid[:,2]))
        # print(pixels_valid.shape)
        # plt.figure(0)
        # plt.hist(pixels_valid[:,0], bins = 256)
        # plt.savefig("0.jpg")
        # plt.figure(1)
        # plt.hist(pixels_valid[:,1], bins = 256)
        # plt.savefig("1.jpg")
        # plt.figure(2)
        # plt.hist(pixels_valid[:,2], bins = 256)
        # plt.savefig("2.jpg")
        # # plt.show()

        
        # print(img.shape)
        x = np.linspace(0, img.shape[0]-1,img.shape[0]) #280(0,279)

        y = np.linspace(0, img.shape[1]-1,img.shape[1]) #387       

        xv, yv = np.meshgrid(y, x)
        # print(xv[1,0],yv[1,0])
        # print('img shape', img.shape, xv.shape, yv.shape)
        
        xv = xv - center[0]
        yv = yv - center[1]

        rv = np.sqrt(xv**2 + yv**2)

     
        # print('radius_p', radius_p, ball_radius_p)
        
        radius_p = min(radius_p, ball_radius_p-1) #取真實半徑和從圖中測量半徑中小的一個作爲範圍(27,22-1)
        mask = (rv < radius_p)   #接觸區域內才爲true的矩陣
        mask_small = (rv < radius_p-1)


        # gradmag=np.arcsin(rv*mask/ball_radius_p)*mask
        # graddir=np.arctan2(-yv*mask, -xv*mask)*mask
        # gradx_img=gradmag*np.cos(graddir)
        # grady_img=gradmag*np.sin(graddir)
        # depth = fast_poisson(gradx_img, grady_img)



        temp = ((xv*mask)**2 + (yv*mask)**2)*self.Pixmm**2 #乘pixmm從像素值變成真實值
     
        height_map = (np.sqrt(self.BallRad**2-temp)*mask - np.sqrt(self.BallRad**2-(radius_p*self.Pixmm)**2))*mask #前面一項是（小於）半球上任一點的高度，後一項是，若沉入的部分小於一個半球，則減去多餘的部分
        
        # np.savetxt('001.txt',height_map)  
        # pdb.set_trace()
        height_map[np.isnan(height_map)] = 0
        # np.savetxt('001.txt',height_map)  
        # pdb.set_trace()
        # cv2.imshow('a',(height_map*100).astype(np.uint8))
        # cv2.waitKey(0)
        # pdb.set_trace()
        # depth = poisson_reconstruct(grady_img, gradx_img, np.zeros(grady_img.shape))


        gx_num = signal.convolve2d(height_map, np.array([[0,0,0],[0.5,0,-0.5],[0,0,0]]), boundary='symm', mode='same')*mask_small #求x梯度
        gy_num = signal.convolve2d(height_map, np.array([[0,0,0],[0.5,0,-0.5],[0,0,0]]).T, boundary='symm', mode='same')*mask_small
        # np.savetxt('002.txt',gx_num)
        # cv2.imshow('a',(gx_num*10).astype(np.uint8))
        # cv2.waitKey(0)
        # pdb.set_trace()

        # depth_num = fast_poisson(gx_num, gy_num)
        # img2show = img.copy().astype(np.float64)
        # img2show[:,:,1] += depth_num*50
        # cv2.imshow('depth_img', img2show.astype(np.uint8))
        # cv2.imshow('valid_mask', valid_mask*255)
        # cv2.waitKey(0)

        gradxseq = gx_num[valid_mask>0]
        gradyseq = gy_num[valid_mask>0]
        # print(gradxseq)
        # pdb.set_trace()
        for i in range(gradxseq.shape[0]):
            b, g, r = pixels_valid[i,0], pixels_valid[i,1], pixels_valid[i,2]
            # print(r,g,b)
            
            if table_account[b,g,r] < 1.: 
                table[b,g,r,0] = gradxseq[i]
                table[b,g,r,1] = gradyseq[i]
                table_account[b,g,r] += 1
            else: #若某個rgb值重復出現一次以上，則把所有的梯度進行平均賦予該rgb
                # print(table[b,g,r,0], gradxseq[i], table[b,g,r,1], gradyseq[i])
                table[b,g,r,0] = (table[b,g,r,0]*table_account[b,g,r] + gradxseq[i])/(table_account[b,g,r]+1) 
                table[b,g,r,1] = (table[b,g,r,1]*table_account[b,g,r] + gradyseq[i])/(table_account[b,g,r]+1)
                table_account[b,g,r] += 1
        
        return table, table_account
        
    

  
    def smooth_table(self, table, count_map):
        y,x,z = np.meshgrid(np.linspace(0,self.bin_num-1,self.bin_num),\
        	np.linspace(0,self.bin_num-1,self.bin_num),np.linspace(0,self.bin_num-1,self.bin_num))
     	
        unfill_x = x[count_map<1].astype(int)
        unfill_y = y[count_map<1].astype(int)
        unfill_z = z[count_map<1].astype(int)
        # print('unfill number', unfill_x.shape)
        fill_x = x[count_map>0].astype(int)
        fill_y = y[count_map>0].astype(int)
        fill_z = z[count_map>0].astype(int)
        # print('unfill number', fill_x.shape)
        fill_gradients = table[fill_x, fill_y, fill_z,:]
        table_new = np.array(table)
        for i in range(unfill_x.shape[0]):
            distance = (unfill_x[i] - fill_x)**2 + (unfill_y[i] - fill_y)**2 + (unfill_z[i] - fill_z)**2
            if np.min(distance) < 20:
	            index = np.argmin(distance)
	           
	            table_new[unfill_x[i], unfill_y[i], unfill_z[i],:] = fill_gradients[index,:]
        
        return table_new
        

         

if __name__=="__main__":
    cali = calibration()
    imp = image_processor()
    padx = 50  #超參
    pady = 10  #超參
    ref_img = cv2.imread('./test_data/ref.jpg')
    # ref_img = color_corre(ref_img)
    ref_img = imp.undistort(ref_img)
    ref_img = imp.crop_image(ref_img, padx, pady)  
    
    # cv2.imshow("1",ref_img)
    # cv2.waitKey(0)
    # pdb.set_trace()
    
    marker = cali.mask_marker(ref_img) #粗糙的markermask  #超參

    keypoints = cali.find_dots(marker) #maker擬合的中心點坐標
    
    marker_mask = cali.make_mask(ref_img, keypoints) #擬合成圓的markermask


    # cv2.imshow("1",marker)
    # cv2.waitKey(0)
    # cv2.imshow("2",marker_mask)
    # cv2.waitKey(0)
    # pdb.set_trace()

    # marker_image = np.dstack((marker_mask,marker_mask,marker_mask)) 
    ref_img = cv2.inpaint(ref_img,marker_mask,3,cv2.INPAINT_TELEA) #將原圖的marker部分去除，即修復圖像，只剩下背景圖。
    
    
    # cv2.imshow("1", ref_img)
    # cv2.waitKey(0)
    # pdb.set_trace()


    table = np.zeros((cali.blue_bin, cali.green_bin, cali.red_bin, 2)) #創建查找表

    table_account = np.zeros((cali.blue_bin, cali.green_bin, cali.red_bin))


    has_marke = True 
    img_list = glob.glob("test_data/s*.jpg")
    
    for name in img_list:
        img = cv2.imread(name)
        # img = color_corre(img)
        img = imp.undistort(img)
        img = imp.crop_image(img, padx, pady)
        # cv2.imshow("1", img)
        if has_marke: 
            marker = cali.mask_marker(img)
            keypoints = cali.find_dots(marker)
            marker_mask = cali.make_mask(img, keypoints) #擬合成圓的markermask,(0or255)
        else:
            marker_mask = np.zeros_like(img) #跟img一樣大的0陣
        # cv2.imshow("2", marker_mask)
        # key = cv2.waitKey(0)
        # if key == ord('q'):
        #     break
        # pdb.set_trace()
        valid_mask, center, radius_p  = cali.contact_detection(img, ref_img, marker_mask) #找到接觸區域輪廓的中心點和半徑，以及接觸區域的mask  #超參

        table, table_account = cali.get_gradient_v2(img, ref_img, center, radius_p, valid_mask, table, table_account)  
 
    np.save('table_0.npy', table)
    np.save('count_map_0.npy', table_account)
    table = np.load('table_0.npy') 
    table_account = np.load('count_map_0.npy') 
    table_smooth = cali.smooth_table(table, table_account)
    np.save('table_0_smooth.npy', table_smooth)
    print('calibration table is generated')
    np.save('count_map_0.npy', table_account)
#%%
#def make_kernal(n,k_type):
#    if k_type == 'circle':
#        kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n))
#    else:
#        kernal = cv2.getStructuringElement(cv2.MORPH_RECT,(n,n))
#    return kernal 
#
#table = np.load('table.npy')
#mask = (np.abs(table[:,:,:,0])>0).astype(np.uint8)
#mask_orginal = mask.copy()
#kernal = make_kernal(9,'circle')
#for i in range(90):
#    mask[:,:,i] = cv2.dilate(mask[:,:,i], kernal, iterations=1)
#    mask[:,:,i] = cv2.erode(mask[:,:,i], kernal, iterations=1)
#
#mask = mask - mask_orginal
#%%




#table_smooth = table.copy()
#
##for j in range(2):
##    for i in range(90):
#
#for i in range(90):
#    table_smooth[:,:,i,0] = inpaint.inpaint_biharmonic(table_smooth[:,:,i,0], \
#                mask[:,:,i],multichannel=False)
#    table_smooth[:,:,i,1] = inpaint.inpaint_biharmonic(table_smooth[:,:,i,1], \
#                mask[:,:,i],multichannel=False)
#    print(i)

   
#%%
#from scipy.interpolate import griddata
#table = np.load('table.npy')
#mask = (np.abs(table[:,:,:,0])>0).astype(np.uint8)
#grid_x, grid_y = np.mgrid[0:90, 0:90]
#table_smooth = table.copy()
#for i in range(90):
#    x_coor = grid_x[mask[:,:,i]>0]
#    y_coor = grid_y[mask[:,:,i]>0]
#    coor_data = np.vstack((x_coor, y_coor)).T
#    datax_temp = table_smooth[:,:,i,0]
#    datay_temp = table_smooth[:,:,i,1]
#    data_x = datax_temp[mask[:,:,i]>0]
#    data_y = datay_temp[mask[:,:,i]>0]
#    
##    print(coor_data.shape)
#    if data_x.shape[0] >0:
#        table_smooth[:,:,i,0] = griddata(coor_data, data_x, (grid_x, grid_y), method='linear')
#        table_smooth[:,:,i,1] = griddata(coor_data, data_y, (grid_x, grid_y), method='linear')
#    print(i/90.)
#    
#    
#
#table_smooth[np.isnan(table_smooth)] = 0.
#%%
    # from mpl_toolkits.mplot3d import Axes3D
    # num = 30
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # X = np.arange(0, 90, 1)
    # Y = np.arange(0, 90, 1)
    # X, Y = np.meshgrid(X, Y)

    # surf = ax.plot_surface(X, Y, table_smooth[num,:,:,0])
    # #plt.figure(0)
    # #plt.imshow(table[:,:,num,0])
    # #plt.figure(1)
    # #plt.imshow(table_smooth[:,:,num,0])
    # #plt.figure(2)
    # #plt.imshow(mask[:,:,num])
    # plt.savefig("a.jpg")
    # plt.show()











# %%
