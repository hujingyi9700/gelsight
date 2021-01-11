import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(dpi=120)
ax = Axes3D(fig)
# Plot 3D Points
depth = np.load("depth.npy")
# x = [k[0] for k in _points]
# y = [k[1] for k in _points]
# z = [k[2] for k in _points]
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# # ax.set_xlim3d(xmin=-15, xmax=25)
# # ax.set_ylim3d(ymin=-20, ymax=20)
# # ax.set_zlim3d(zmin=0, zmax=50)
# ax.scatter(x, y, z, c='r', marker='.', s=5, linewidth=2, alpha=1, cmap='spectral')
# plt.pause(0.01)
# plt.cla()



# ax.set_aspect('equal')
X = np.arange(0, depth.shape[1], 1)*0.05
Y = np.arange(0, depth.shape[0], 1)*0.05
X, Y = np.meshgrid(X, Y)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# ax.set_xlim3d(xmin=0, xmax=30)
# ax.set_ylim3d(ymin=0, ymax=30)
ax.set_zlim3d(zmin=0, zmax=10)
surf = ax.plot_surface(X, Y, depth, cmap=plt.get_cmap('rainbow'))
# ax.contourf(X,Y,Z,zdir='z',offset=-2,cmap='rainbow')
plt.show()


# frame = np.uint8(np.clip((1.5 * frame + 25), 0, 255))
# for i in range(9):
#     cv2.line(frame, (0, 48 * (i + 1)), (1280, 48 * (i + 1)), (0, 0, 255), 2)
#     cv2.line(frame_rec, (0, 48 * (i + 1)), (1280, 48 * (i + 1)), (0, 0, 255), 2)
# cv2.imshow('GelStereo_Original_Frame', frame)


