from yolox.data.data_augment import *
import cv2
import time

import numpy as np
import cv2

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

img = cv2.imread("/data1/qilei_chen/DEVELOPMENTS/YOLOX/datasets/gastric_object_detection/images/0_03_00.700000.jpg")


res_img,_ = random_perspective(img,scale=[0.95,1.05],shear=0,translate=0.0)
cv2.imwrite("/data2/qilei_chen/DATA/new_polyp_data_combination/testrp.jpg",res_img)
#cv2.imwrite("/data2/qilei_chen/DATA/new_polyp_data_combination/test.jpg",rotate_image(img,25))
