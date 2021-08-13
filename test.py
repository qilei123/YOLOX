import cv2 
import matplotlib.pyplot as plt 
 
img = cv2.imread('/data2/qilei_chen/DATA/trans_drone/videos/results1/test.jpg') 
rows,cols = img.shape[:2] 
#第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例 
M = cv2.getRotationMatrix2D((cols/2,rows/2),45,1) 
#第三个参数：变换后的图像大小 
res = cv2.warpAffine(img,M,(rows,cols)) 

cv2.imwrite("/data2/qilei_chen/DATA/trans_drone/videos/results1/test1.jpg",res)

