import cv2
import numpy as np

maps = np.zeros((100,100))
cv2.circle(maps, ( 50, 50 ), int(8), 1, -1)

cv2.imshow("circle", maps)
cv2.waitKey()