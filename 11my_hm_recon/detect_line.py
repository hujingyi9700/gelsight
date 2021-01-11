import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

depth = np.load("depth.npy")
print(np.max(depth))
img = normalization(depth)
img = img*255
img = img.astype(np.uint8)
img = cv2.GaussianBlur(img, (7, 7), 0)

# plt.hist(img.ravel(),256)
# plt.show()

#easy_threshold

ret, binary = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)  
contours, _ = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
# cv2.drawContours(img, contours, -1, (0,0,255), 3)  
# print(contours)
cnt = contours[0]
ellipse = cv2.fitEllipse(cnt)

img = cv2.ellipse(img,ellipse,(0,255,0),2)

#adaptive效果不好，會得到比較碎的輪廓
# 
# th = cv2.adaptiveThreshold(img, 255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,51,5)
# contours, hierarchy = cv2.findContours(th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
# cv2.drawContours(img, contours, -1,(0,0,255), 3)  

#Otsu 滤波
# 
# ret2,th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# contours, hierarchy = cv2.findContours(th2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
# cv2.drawContours(img, contours, -1,(0,0,255), 3) 


cv2.imshow("img", img)  
cv2.waitKey(0)  
#canny算子效果不好
# edges = cv2.Canny(img,10, 30)

# plt.subplot(121),plt.imshow(img,cmap='gray')
# plt.title('original'),plt.xticks([]),plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap='gray')
# plt.title('edge'),plt.xticks([]),plt.yticks([])

# plt.show()