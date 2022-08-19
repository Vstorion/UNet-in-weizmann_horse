import numpy as np
import cv2

# size=(H,W)
def fast_hist(a, b, n=2):
    # a = label.flatten();b = pred.flatten();n is a real
    k = (a >= 0) & (a < n)  # k是一个一维bool数组，形状(H×W,)；目的是找出标签中需要计算的类别（去掉了背景）
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n,\
                          n)  # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)

def per_class_iu(hist):  # 分别为每个类别计算mIoU，hist的形状(n, n)
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(
        hist))  # 矩阵的对角线上的值组成的一维数组/矩阵的所有元素之和，返回值形状(n,)

def mIoU(a,b):
    # a = label.flatten();b = pred.flatten();
    hist = fast_hist(a,b)
    result = np.mean(per_class_iu(hist))
    return result

def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode

def Boundary_IoU(mask,pred):
    Gd = mask_to_boundary(mask)
    G = mask
    Pd = mask_to_boundary(pred)
    P = pred
    Gd_cap_G = [val for val in Gd if val in G]
    Pd_cap_P = [val for val in Pd if val in P]
    BIoU = mIoU(np.array(Gd_cap_G),np.array(Pd_cap_P))
    return BIoU
    
