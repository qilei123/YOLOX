from yolox.data.data_augment import *
import cv2

img = cv2.imread("/data1/qilei_chen/DEVELOPMENTS/YOLOX/datasets/gastric_object_detection/images/0_03_00.700000.jpg")

res_img,_ = random_perspective(img,scale=[0.1,1.5])

cv2.imwrite("datasets/test1.jpg",res_img)