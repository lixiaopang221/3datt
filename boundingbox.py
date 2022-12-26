import numpy as np
import nibabel as nib
import os
from skimage import measure
from scipy.ndimage.morphology import binary_dilation, binary_erosion


def get_path(sample_name):
    path = './result/prevalidation/'
    seg_names = os.listdir(path)
    for name in seg_names:
        if sample_name in name:
            return path + name
    
def conneted_component(img):
    img[img > 0] = 1
    img_erosion = binary_erosion(img, iterations = 3)
    labels = measure.label(img_erosion)
    img_area = []
    for i in range(1, np.max(labels) + 1):
        img_temp = np.zeros_like(img)
        img_temp[labels == i] = 1
        img_area.append(img_temp)
    return img_area
    
def get_box_points(sample_name):
    img_path = get_path(sample_name)
    img = nib.load(img_path).get_data()
    points = []
    for comp in conneted_component(img):
        nzls = np.nonzero(comp)
        max_p = np.max(nzls, axis = 1)
        min_p = np.min(nzls, axis = 1)
        p = min_p + (max_p - min_p)/2 - 48
        p = np.stack(p).astype('int8')
        p_limit_min = np.array([20,20,0])
        p_limit_max = np.array([124,124,56])
        p[p < p_limit_min] = p_limit_min[p < p_limit_min]
        p[p > p_limit_max] = p_limit_max[p > p_limit_max]
        points.append(p)
    return np.stack(points)
    
def set_ps(patcher, sample_name):
    points = get_box_points(sample_name)
    points = points[:,[2,1,0]]
    patcher.points = points[:3]
    return patcher
    
if __name__ == '__main__':
    with open('../../data/validation.txt', 'r') as file:
        samples = file.read().splitlines()
    for sample in samples:
        p = get_box_points(sample)
        print(p)