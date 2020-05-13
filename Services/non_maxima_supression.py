import cv2 as cv
import config as cfg
import numpy as np

from Models.boxModel import BoxModel, ResultBoxModel, BoxesWithEthalons

def start_nms(model: ResultBoxModel, scale: int):

    ethalons = compute_ethalons(model,scale*10)
    most_accurate_boxes = []
    
    for eth in ethalons:
        boxes = eth.boxes
        confidence = 0
        max_box = BoxModel()
        for box in boxes:
            if box.confidence > confidence:
                confidence = box.confidence
                max_box = box
        
        most_accurate_boxes.append(max_box)
        
    output = ResultBoxModel()
    output.boxes = most_accurate_boxes
    output.file_name = model.file_name
                
    
    return output

    # img = cv.imread(r'C:\Users\Lukas\Desktop\semestralny projekt\test_nms.png', 0)

    # print("running non maxima suppresion algorithm")
    # # removed values (pixels) with value lower than treshold value
    # # img = [0 if tresholded_ < treshold for tresholded_ in img]
    # cv.imshow("test image for non maxima suppresion", img)
    # thesh_indices = img < treshold
    # img[thesh_indices] = 0

    # maximums = []
    # for y in range(img.shape[0]):
    #     for x in range(img.shape[1]):

    #         if img[y][x] != 0:
    #             max_y, max_x, value = find_local_maximum(y, x, img, img[y][x], y, x )
    #             maximums.append([max_y, max_x, value])

    # # for i in range(len(maximums)):
    # #     cv.circle(img,[maximums[i][0],maximums[i][1]],1,[255,0,0])

    # cv.imshow("test image for non maxima suppresion", img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # print(maximums)

def compute_ethalons(model : ResultBoxModel, deviation):
    
    ethalons = []
    for box in model.boxes:
        top_right = find_top_right_corner(box)
        bottom_left = find_bottom_left_corner(box)
        
        center = ((top_right[0] + bottom_left[0]) /2, (top_right[1] + bottom_left[1]) /2)
        if len(ethalons) == 0:
            eth = BoxesWithEthalons()
            eth.ethalon = center
            eth.boxes.append(box)
            ethalons.append(eth)
            
        else:
            lowest_ethalon_index = 0
            min_dist = 10000            
            for i in range(len(ethalons)):
                distance = compute_euclidean(ethalons[i].ethalon,center)
                if distance < deviation and distance < min_dist:
                   lowest_ethalon_index = i
                   min_dist = distance
            
            if min_dist != 10000: # ethalon elrady exists
                ethalons[lowest_ethalon_index].boxes.append(box)
            else:
                eth = BoxesWithEthalons()
                eth.ethalon = center
                eth.boxes.append(box)
                ethalons.append(eth)
                
    return ethalons
        
def compute_euclidean(p1, p2):
    distance = np.sqrt( pow(p2[0] -p1[0],2) + pow(p2[1] - p1[1],2) )
    return distance;    

def find_top_right_corner(box: BoxModel):
    top_right = box.fbl
    if box.fbr[0] > top_right[0] and box.fbr[1] < top_right[1]: # x,y
        top_right = box.fbr
    if box.rbl[0] > top_right[0] and box.rbl[1] < top_right[1]: # x,y
        top_right = box.rbl
    if box.rbr[0] > top_right[0] and box.rbr[1] < top_right[1]: # x,y
        top_right = box.rbr
    if box.ftl[0] > top_right[0] and box.ftl[1] < top_right[1]: # x,y
        top_right = box.ftl
    if box.ftr[0] > top_right[0] and box.ftr[1] < top_right[1]: # x,y
        top_right = box.ftr
    if box.rtl[0] > top_right[0] and box.rtl[1] < top_right[1]: # x,y
        top_right = box.rtl
    if box.rtr[0] > top_right[0] and box.rtr[1] < top_right[1]: # x,y
        top_right = box.rtr

    return top_right
    
def find_bottom_left_corner(box: BoxModel):
    bottom_left = box.fbl
    if box.fbr[0] < bottom_left[0] and box.fbr[1] > bottom_left[1]: # x,y
        bottom_left = box.fbr
    if box.rbl[0] < bottom_left[0] and box.rbl[1] > bottom_left[1]: # x,y
        bottom_left = box.rbl
    if box.rbr[0] < bottom_left[0] and box.rbr[1] > bottom_left[1]: # x,y
        bottom_left = box.rbr
    if box.ftl[0] < bottom_left[0] and box.ftl[1] > bottom_left[1]: # x,y
        bottom_left = box.ftl
    if box.ftr[0] < bottom_left[0] and box.ftr[1] > bottom_left[1]: # x,y
        bottom_left = box.ftr
    if box.rtl[0] < bottom_left[0] and box.rtl[1] > bottom_left[1]: # x,y
        bottom_left = box.rtl
    if box.rtr[0] < bottom_left[0] and box.rtr[1] > bottom_left[1]: # x,y
        bottom_left = box.rtr

    return bottom_left


def find_local_maximum(y, x, img, last_max, max_y, max_x):
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
    for spaceing in range(1, 100, 1):

        treshhold_area = True
        max_has_changed = True
        while max_has_changed:
            max_has_changed = False
            for tmp_y in range(max_y-spaceing, max_y + 2*spaceing + 1, 1):
                # check vertical lines of pixels
                # out of bounds
                if tmp_y < 0 or tmp_y >= img.shape[0] or max_x-spaceing < 0 or max_x+spaceing >= img.shape[1]:
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

            for tmp_x in range(max_x-spaceing, max_x+2*spaceing + 1, 1):
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
