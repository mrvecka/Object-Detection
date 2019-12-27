import cv2 as cv
import config as cfg

# Finding peak element in a 2D Array. 
treshold = cfg.NMS_TRESHOLD

def start_NMS():
    img = cv.imread(r'C:\Users\Lukas\Desktop\semestralny projekt\test_nms.png', 0)
    
    print("running non maxima suppresion algorithm")
    # removed values (pixels) with value lower than treshold value
    # img = [0 if tresholded_ < treshold for tresholded_ in img]
    cv.imshow("test image for non maxima suppresion", img)
    thesh_indices = img < treshold
    img[thesh_indices] = 0
    
    maximums = []
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            
            if img[y][x] != 0:
                max_y, max_x, value = find_local_maximum(y, x, img, img[y][x], y, x )
                maximums.append([max_y, max_x, value])
        
    # for i in range(len(maximums)):
    #     cv.circle(img,[maximums[i][0],maximums[i][1]],1,[255,0,0])
           
    cv.imshow("test image for non maxima suppresion", img)
    cv.waitKey(0)
    cv.destroyAllWindows()     
    print(maximums)

def find_local_maximum(y,x,img, last_max, max_y, max_x):
    """
    Find local maximum if image. Starting point is (x,y) and algorithm looks all posible ways
    Values other than maximum will be set to 0

    Input:
        x: x coordinate of starting point
        y: y coordinate of starting point
        img: processed image
    Returns:
        Point where maximum is presented
    """
    
    if x == 147 and y == 156:
        cv.imshow(img)
        cv.waitKey(0) 
    
    last_max = img[y][x]
    max_y = y    
    max_x = x
    
        # * * *
        # * x *
        # * * *  
    for spaceing in range(1,100,1):
        
        treshhold_area = True
        max_has_changed = True
        while max_has_changed:
            max_has_changed = False
            for tmp_y in range(max_y-spaceing, max_y+ 2*spaceing + 1,1):
                # check vertical lines of pixels
                if tmp_y < 0 or tmp_y >= img.shape[0] or max_x-spaceing < 0 or max_x+spaceing >= img.shape[1]: # out of bounds
                    continue
                
                if img[tmp_y][max_x-spaceing] != 0:
                    treshhold_area = False
                    
                if img[tmp_y][max_x-spaceing] > last_max:
                    last_max = img[tmp_y][max_x-spaceing]
                    max_y = tmp_y
                    max_x = max_x-spaceing
                    max_has_changed = True
                    break
                else:
                    img[tmp_y][max_x-spaceing] = 0

                if img[tmp_y][max_x+spaceing] != 0:
                    treshhold_area = False
                    
                if img[tmp_y][max_x+spaceing] > last_max:
                    last_max = img[tmp_y][max_x+spaceing]
                    max_y = tmp_y
                    max_x = max_x+spaceing
                    max_has_changed = True
                    break
                else:
                    img[tmp_y][max_x+spaceing] = 0
                    
            for tmp_x in range(max_x-spaceing,max_x+2*spaceing + 1,1):
                # check horizontal lines of pixels
                if tmp_x < 0 or tmp_x >= img.shape[1] or max_y-spaceing < 0 or max_y+spaceing >= img.shape[0]:
                    continue
                
                if img[max_y-spaceing][tmp_x] != 0:
                    treshhold_area = False
                    
                if img[max_y-spaceing][tmp_x] > last_max:
                    last_max = img[max_y-spaceing][tmp_x]
                    max_y = max_y-spaceing
                    max_x = tmp_x
                    max_has_changed = True
                    break
                else:
                    img[max_y-spaceing][tmp_x] = 0

                if img[max_y+spaceing][tmp_x] != 0:
                    treshhold_area = False
                    
                if img[max_y+spaceing][tmp_x] > last_max:
                    last_max = img[max_y+spaceing][tmp_x]
                    max_y = max_y+spaceing
                    max_x = tmp_x
                    max_has_changed = True
                    break
                else:
                    img[max_y+spaceing][tmp_x] = 0
                
                
            
        if treshhold_area:
            break
    
    return max_y, max_x, last_max

if __name__ == '__main__':
    start_NMS()