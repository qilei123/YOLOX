from yolox.data.data_augment import *
import cv2
import time

img = cv2.imread("/data1/qilei_chen/DEVELOPMENTS/YOLOX/datasets/gastric_object_detection/images/0_03_00.700000.jpg")

res_img,_ = random_perspective(img,scale=[0.1,2],shear=2)

cv2.imwrite("datasets/"+str(time.time())+".jpg",res_img)