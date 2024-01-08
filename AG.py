import cv2
import torch
import os
import collections
from collections import Counter
import math
import numpy as np
import math
from PIL import Image
from tqdm import tqdm


#img = cv2.imread('./PyTorch/data/car/1/13.jpg') #读取图像




'''
def spatialF(path):
    image = Image.open(path).convert('L')
    image = np.array(image)
    M = image.shape[0]
    N = image.shape[1]

    cf = 0
    rf = 0
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            dx = float(image[i, j - 1]) - float(image[i, j])
            rf += dx ** 2
            dy = float(image[i - 1, j]) - float(image[i, j])
            cf += dy ** 2

    RF = math.sqrt(rf / (M * N))
    CF = math.sqrt(cf / (M * N))
    SF = math.sqrt(RF ** 2 + CF ** 2)

    return round(SF,2)

if __name__ == "__main__":
    for root, dirs, files in os.walk('./PyTorch/data/car/1/'):
        for i in files:
        # print(i)'
            #print(str(i) + ':' + str(spatialF('./PyTorch/data/car/2/' + str(i))))
            print(str(spatialF('./PyTorch/data/car/1/' + str(i))))'''

'''
image_dir = './PyTorch/data/sesr/1/9.jpg'
img = cv2.imread(image_dir, flags=cv2.IMREAD_GRAYSCALE)
img = torch.from_numpy(img)
compare_list = []
for m in range(1, img.size()[0] - 1):
    for n in range(1, img.size()[0] - 1):
        sum_element = img[m - 1, n - 1] + img[m - 1, n] + img[m - 1, n + 1] + img[m, n - 1] + img[m, n + 1] + img[
            m + 1, n - 1] + img[m + 1, n] + img[m + 1, n + 1]
        sum_element = int(sum_element)
        mean_element = sum_element // 8
        pix = int(img[m, n])
        temp = (pix, mean_element)
        compare_list.append(temp)

#print(compare_list)
compare_dict = collections.Counter(compare_list)
H = 0.0
for freq in compare_dict.values():
    f_n2 = freq / img.size()[0] ** 2
    log_f_n2 = math.log(f_n2)
    h = -(f_n2 * log_f_n2)
    H += h

print(H)'''

'''
if __name__ == "__main__":
    for root, dirs, files in os.walk('./PyTorch/data/car/1/'):
        for i in files:
        # print(i)
            print(str(i)+':'+str(avgGradient('./PyTorch/data/car/1/' + str(i))))'''

