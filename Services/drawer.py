import cv2

def draw_bounding_boxes(image_model):
    
    img = cv2.imread(image_model.filename, cv2.IMREAD_COLOR)
    
    for i in range(len(image_model.boxes)):
        box = image_model.boxes[i]
        
        # front
        cv2.line(img, box.fbl, box.fbr, (0,0,255), 2) 
        cv2.line(img, box.fbr, box.ftr, (0,0,255), 2) 
        cv2.line(img, box.ftr, box.ftl, (0,0,255), 2) 
        cv2.line(img, box.ftl, box.fbl, (0,0,255), 2)
         
        # rear
        cv2.line(img, box.rbl, box.rbr, (255,0,0), 2) 
        cv2.line(img, box.rbr, box.rtr, (255,0,0), 2) 
        cv2.line(img, box.rtr, box.rtl, (255,0,0), 2) 
        cv2.line(img, box.rtl, box.rbl, (255,0,0), 2) 
        
        # connections
        cv2.line(img, box.fbl, box.rbl, (51,51,51), 2)
        cv2.line(img, box.fbr, box.rbr, (51,51,51), 2)
        cv2.line(img, box.ftl, box.rtl, (51,51,51), 2)
        cv2.line(img, box.ftr, box.rtr, (51,51,51), 2)
        
    
    cv2.imshow("result without NMS", img)
    cv2.waitKey(0)
        
        
    
    