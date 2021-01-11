import cv2
import numpy as np
 
def gamma_trans(img, gamma):  # gamma函数处理
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
    return cv2.LUT(img, gamma_table)  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。

def satura_bright(image,l,s):
    # 加载图片 读取彩色图像
    # image = cv2.imread('test_data/s1.jpg', cv2.IMREAD_COLOR)
    # print(image)
    # cv2.imshow("image", image)
    # 图像归一化，且转换为浮点型
    fImg = image.astype(np.float32)
    fImg = fImg / 255.0
    # 颜色空间转换 BGR转为HLS
    hlsImg = cv2.cvtColor(fImg, cv2.COLOR_BGR2HLS)

    MAX_VALUE = 100
    # 调整饱和度和亮度后的效果
    lsImg = np.zeros(image.shape, np.float32)
    # 调整饱和度和亮度
        # 复制
    hlsCopy = np.copy(hlsImg)
        # 1.调整亮度（线性变换) , 2.将hlsCopy[:, :, 1]和hlsCopy[:, :, 2]中大于1的全部截取
    hlsCopy[:, :, 1] = (1.0 + l / float(MAX_VALUE)) * hlsCopy[:, :, 1]
    hlsCopy[:, :, 1][hlsCopy[:, :, 1] > 1] = 1
        # 饱和度
    hlsCopy[:, :, 2] = (1.0 + s / float(MAX_VALUE)) * hlsCopy[:, :, 2]
    hlsCopy[:, :, 2][hlsCopy[:, :, 2] > 1] = 1
        # HLS2BGR
    lsImg = cv2.cvtColor(hlsCopy, cv2.COLOR_HLS2BGR)

    lsImg = lsImg * 255
    lsImg = lsImg.astype(np.uint8)
    return lsImg

def color_corre(image):
    #加载图像
    # image = cv2.imread('test_data/16.jpg')
    # img = contrast_demo(image, 1, 5)
    l = 3
    s = 30
    img = satura_bright(image,l,s)

    gamma_val = 1.5#math.log10(0.5)/math.log10(mean/255)    # 公式计算gamma
    img = gamma_trans(img, gamma_val)   # gamma变换
    #自定义卷积核
    # kernel_sharpen_1 = np.array([
    #         [-1,-1,-1],
    #         [-1,9,-1],
    #         [-1,-1,-1]])
    kernel_sharpen_1 = np.array([
            [0,-1,0],
            [-1,5,-1],
            [0,-1,0]])
    kernel_sharpen_3 = np.array([
            [-1,-1,-1,-1,-1],
            [-1,2,2,2,-1],
            [-1,2,8,2,-1],
            [-1,2,2,2,-1], 
            [-1,-1,-1,-1,-1]])/8.0
    #卷积
    output_1 = cv2.filter2D(img,-1,kernel_sharpen_1)
    return output_1
