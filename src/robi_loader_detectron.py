import os,sys
import torch
import detectron2
from detectron2.structures import BoxMode
import  inout
import cv2
import numpy as np



def robiLoader(obj_name,class_id,DataPath):
    """
    Function to load the ROBI dataset for Detectron2.
    Returns a list of dictionaries, each containing image and annotation data. 
    This is only synthetic data.
    Args:
    (str)obj_name - the name of the object to load from the ROBI dataset eg: "Chrome_screw_synthetic" 
    (dict)class_to_id_mapping - a mapping from class names to class IDs
    The required format is:
    {
    'file_name': 'path/to/your/image.png',
    'image_id': 'some_unique_id',
    'height': 1024,
    'width': 1280,
    'annotations': [
        {
            'bbox': [x, y, w, h],  # Bounding box
            'bbox_mode': BoxMode.XYWH_ABS,
            'segmentation': [[x1, y1, x2, y2, ...]],  # Segmentation polygon
            'category_id': 0, # The class of the object
        },
        # ... more annotations for other objects
        ]
    }
    """

    dataset_dicts = [] #stores list of dicts
    #define directories
    scene_dir = os.path.join(DataPath,obj_name)
    print(f"Loading scene from {scene_dir}")
    if not os.path.exists(scene_dir):
        print(f"Directory {scene_dir} does not exist.")
        return []

    for json_file in os.listdir(scene_dir):
        #print("starting " + json_file)
        #only apply operation to each instance of json file. this prevents repeated operations as naming is consistent (Except for suffix)
        if not json_file.endswith(".json"):
            continue
        
        json_path = os.path.join(scene_dir, json_file)
        depth_path = os.path.join(scene_dir, json_file.replace(".json", "_Depth.png"))
        left_path = os.path.join(scene_dir, json_file.replace(".json", "_Left.png"))
        mask_path = os.path.join(scene_dir, json_file.replace(".json", "_Mask.png"))

        record = {}
        record["file_name"] = left_path
        record["image_id"] = json_file.replace(".json", "")
        record["height"] = cv2.imread(left_path).shape[0]
        record["width"] = cv2.imread(left_path).shape[1]

        scene_info = inout.load_sceneInfo(json_path) #loads scene information (poses, visibility, intrinsics)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)[:,:,2] #read the mask image. each object's pixel values matches the object ID
    
        num_obj = len(scene_info)
        annotations = [] #stores annotations for each object in the image
        for obj_idx in range(2, num_obj + 1):
            obj_name_idx = f"Object_{obj_idx}" #Labels object name

            mask_id = int(scene_info[obj_name_idx]["mask_id"]) #get mask ID
            obj_mask = (mask_img == mask_id).astype(np.uint8)#create binary mask from mask image
            contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)  # Get the largest contour (avoids noise)
            if contour.shape[0]<3:
                continue #skip if less than 3 points (invalid polygon)
            x, y, w, h = cv2.boundingRect(contour)

            segmentation = [contour.flatten().tolist()]  # Flatten the contour to match the expected format

            ann = {                
                'bbox': [x, y, w, h],
                'bbox_mode': BoxMode.XYWH_ABS,
                'segmentation': segmentation,
                'category_id': class_id,
            }
            annotations.append(ann) #append the annotations for this object to the annotations list

        record["annotations"]= annotations #store all annotations for this image
        dataset_dicts.append(record) #append this scene record to the dataset

    return dataset_dicts


