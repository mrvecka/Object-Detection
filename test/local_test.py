import cv2
import numpy as np


img = cv2.imread(r"C:\Users\Lukas\Documents\Object detection 2.0\result_test_s8\response_map_0.jpg", cv2.IMREAD_GRAYSCALE)

for y in range(len(img)):
    for x in range(len(img[0])):
        if img[y,x] != 0:
            x,y = find_local_max(img, x, y)
            print("X: ",x, " Y: ",y)
            
            
def find_local_max(img, x, y):
    loc_y = y -1
    if y-1 < 0: 
        loc_y = y
    if y+1 > img.shape[0]:
        loc_y -= 2 # local area is 3x3
        
    loc_x = x -1
    if x-1 < 0: 
        loc_x = x
    if x+1 > img.shape[1]:
        loc_x -= 2
        
    local_area = img[loc_y:loc_y+3,loc_x:loc_x+3]
    local_max = np.max(local_area)
    while local_max != img[y,x]:
        max_index_col = np.argmax(local_area, axis=0)
        max_index_row = np.argmax(local_area, axis=1)
    
        new_x = max_index_col - 1
        new_y = max_index_row - 1
        
        x = x + new_x
        y = y + new_y
        
        loc_y = y -1
        if y-1 < 0: 
            loc_y = y
        if y+1 > img.shape[0]:
            loc_y -= 2
            
        loc_x = x -1
        if x-1 < 0: 
            loc_x = x
        if x+1 > img.shape[1]:
            loc_x -= 2
        
        local_area = img[loc_y:loc_y+3,loc_x:loc_x+3]
        local_max = np.max(local_area)
        
    return x, y
