__date__   = '14/05/2020'
__author__ = 'Lukas Mrvecka'
__email__  = 'lukas.mrvecka.st@vsb.cz'

import cv2

def draw_bounding_boxes(image_model, scale):
    
    img = cv2.imread(image_model.file_name, cv2.IMREAD_COLOR)
    height = img.shape[0]
    width = img.shape[1]
    resized= img
    
    for i in range(len(image_model.boxes)):
        box = image_model.boxes[i]
        
        # front
        cv2.line(resized, box.fbl, box.fbr, (0,0,255), 2) 
        cv2.line(resized, box.fbr, box.ftr, (0,0,255), 2) 
        cv2.line(resized, box.ftr, box.ftl, (0,0,255), 2) 
        cv2.line(resized, box.ftl, box.fbl, (0,0,255), 2)
         
        # rear
        cv2.line(resized, box.rbl, box.rbr, (255,0,0), 2) 
        cv2.line(resized, box.rbr, box.rtr, (255,0,0), 2) 
        cv2.line(resized, box.rtr, box.rtl, (255,0,0), 2) 
        cv2.line(resized, box.rtl, box.rbl, (255,0,0), 2) 
        
        # connections
        cv2.line(resized, box.fbl, box.rbl, (51,51,51), 2)
        cv2.line(resized, box.fbr, box.rbr, (51,51,51), 2)
        cv2.line(resized, box.ftl, box.rtl, (51,51,51), 2)
        cv2.line(resized, box.ftr, box.rtr, (51,51,51), 2)
        
    resized_back = cv2.resize(resized, (width,height))
    cv2.imshow("result without NMS", resized)
    cv2.imwrite(r"C:\Users\Lukas\Documents\Object detection\output\output_s"+ str(scale) +".jpg",resized)
    cv2.waitKey(0)
        
        
    
    