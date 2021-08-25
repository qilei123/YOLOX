from yolox.data.data_augment import *
import cv2

img = cv2.imread("/data1/qilei_chen/DEVELOPMENTS/YOLOX/datasets/gastric_object_detection/images/0_03_00.700000.jpg")

res_img = random_perspective(img)

cv2.imwrite("datasets/test.jpg",res_img)