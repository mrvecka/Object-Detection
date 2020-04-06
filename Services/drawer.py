import cv2
import Services.fileworker as fw

def draw_bounding_boxes(image_model, scale):
    
    img = cv2.imread(image_model.file_name, cv2.IMREAD_COLOR)
    height = img.shape[0]
    width = img.shape[1]
    resized= img
    
    for i in range(len(image_model.boxes)):
        box = image_model.boxes[i]
        
        color_front = (0,0,255)
        color_back = (255,0,0)
        color_connections = (51,51,51)
        
        if box.confidence == 100:
            color_front = (255,255,255)
            color_back = (255,255,255)
            color_connections = (255,255,255)
        
        # front
        cv2.line(resized, box.fbl, box.fbr, color_front, 2) 
        cv2.line(resized, box.fbr, box.ftr, color_front, 2) 
        cv2.line(resized, box.ftr, box.ftl, color_front, 2) 
        cv2.line(resized, box.ftl, box.fbl, color_front, 2)
         
        # rear
        cv2.line(resized, box.rbl, box.rbr, color_back, 2) 
        cv2.line(resized, box.rbr, box.rtr, color_back, 2) 
        cv2.line(resized, box.rtr, box.rtl, color_back, 2) 
        cv2.line(resized, box.rtl, box.rbl, color_back, 2) 
        
        # connections
        cv2.line(resized, box.fbl, box.rbl, color_connections, 2)
        cv2.line(resized, box.fbr, box.rbr, color_connections, 2)
        cv2.line(resized, box.ftl, box.rtl, color_connections, 2)
        cv2.line(resized, box.ftr, box.rtr, color_connections, 2)
        
    resized_back = cv2.resize(resized, (width,height))
    cv2.imshow("result without NMS", resized)
    
        
    cv2.waitKey(0)
    base_path = r".\output"
    if not fw.check_and_create_folder(base_path):
        return
    cv2.imwrite(base_path + r"\output_s" + str(scale)+".jpg",resized)
        
        
    
    