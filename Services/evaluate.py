from Models.boxModel import ResultBoxModel
import numpy as np

def center_from_X_3x8(X_3x8):
    """
    Computes the center of the 3D bounding box with corners (FBL FBR RBR RBL FTL FTR RTR RTL) in
    the X_3x8 matrix.

    Input:
        X_3x8: matrix with 3D coordinates of a 3D bounding box corners ordered (FBL FBR RBL RBR FTL
                FTR RTL RTR)
    Returns:
        BBC_3x1 matrix with coordinates of the center
    """

    BBC_3x1 = (X_3x8[:,0] + X_3x8[:,7]) / 2.0
    return BBC_3x1

def compute_center_error(gt, predicted):
    """
    Computes the error in the bounding box center position.

    Input:
        gt:  BB3D object with ground truth
        predicted: BB3D object with the detection
    Returns:
        error
    """

    # Coordinates of the bounding box centers
    X_gt_C_3x1  = center_from_X_3x8(gt)
    X_det_C_3x1 = center_from_X_3x8(predicted)

    error = np.linalg.norm(X_gt_C_3x1 - X_det_C_3x1)

    return error

def compute_image_points_error(gt, predicted):
    """
	Computes the error of bounding box corner points in image coordinates.

	Input:
		gt:  BB3D object with ground truth
		predicted: BB3D object with the detection
	Returns:
		error
	"""
    error = np.linalg.norm(gt - predicted)
    return error

def evaluate(boxes:ResultBoxModel):
    box_from_label = list(filter(lambda b: (b.confidence == 100),boxes.boxes))
    predicted = list(filter(lambda b: (b.confidence != 100),boxes.boxes))
    
    global_error = 0
    box_count = 0
    min_error = 10000
    
    for box in box_from_label:
        
        min_error = 10000
        closest_box = None
        for pred_box in predicted:
            center_error = compute_center_error(box.world_points,pred_box.world_points)
            if center_error < min_error:
                closest_box = pred_box
        
        if closest_box != None:
            points_error = compute_image_points_error(box.image_points,closest_box.image_points)
            global_error += points_error
            box_count +=1
            
    try:
        evaluation_error = global_error / box_count
    except:
        evaluation_error = -1
    print("EVALUATION ERROR ON",boxes.file_name,": ",evaluation_error)
            
            
         
        
    