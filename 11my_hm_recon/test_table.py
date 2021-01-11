import cv2
import numpy as np 
import matplotlib.pyplot as plt  
import glob
from fast_poisson import fast_poisson
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from skimage.restoration import inpaint




table = np.load('table_0.npy')
table_smooth = np.load('table_0_smooth.npy') 
count_map = np.load('count_map_0.npy')

print('table size', table.shape)
print('countmap size', count_map.shape)

plt.figure(0)
sc = plt.imshow(table[:,:,45,0])
plt.colorbar(sc)
plt.savefig("x1.jpg")

plt.figure(1)
sc = plt.imshow(table_smooth[:,:,45,0])
plt.colorbar(sc)
plt.savefig("x2.jpg")

plt.figure(2)
sc = plt.imshow(count_map[:,:,45])
plt.colorbar(sc)
plt.savefig("x3.jpg")
# plt.show()
