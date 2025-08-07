import detectron2 as d2 
import torch as T 
import robi_loader_detectron
import numpy as np
import os
import cv2,json,random,os
from detectron2.utils.visualizer import Visualizer
import argparse
import tensorboard
import pickle

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo


#register dataset
ROBI_CLASSES = ["Chrome_screw_synthetic","DIN_connector_synthetic","DSub_connector_synthetic","Eye_bolt_synthetic","Gear_synthetic","Tube_fitting_synthetic","Zigzag_synthetic"]
CLASS_TO_ID = {cls: idx for idx, cls in enumerate(ROBI_CLASSES)}

DataPath = "/home/kaelin/Documents/BinPicking/SegmentationModel/Data/"
pickle_file = "SegmentationModel/Data/robi_dataset.pkl"

# def registerROBIDataset(DataPath, CLASS_TO_ID):
#     for class_name in ROBI_CLASSES:
#         dataset_name = f"robi_{class_name}_train"
    
#     # Use a lambda function to pass arguments to the data loader
#         DatasetCatalog.register(dataset_name, lambda d=class_name: robi_loader_detectron.robiLoader(d, CLASS_TO_ID[d], DataPath=DataPath))
#         MetadataCatalog.get(dataset_name).set(thing_classes=ROBI_CLASSES)

# def registerROBISingleClassDataset(DataPath, CLASS_TO_ID, class_name):
#     dataset_name = f"robi_{class_name}_train"
#     DatasetCatalog.register(dataset_name, lambda d=class_name: robi_loader_detectron.robiLoader(d, CLASS_TO_ID[d], DataPath=DataPath))
#     MetadataCatalog.get(dataset_name).set(thing_classes=ROBI_CLASSES)

def main():
    
    
    parser = argparse.ArgumentParser(
        prog="train_maskrcnn",
        description="Train Mask R-CNN on ROBI dataset using Detectron2"
    )
    parser_setup(parser)
    args=parser.parse_args()

    #load pre-processed pickle
    all_datasets = loadPickle(pickle_file)

    single_dataset_to_train = all_datasets[args.training_class]

    #registerROBISingleClassDataset(args.data_path, CLASS_TO_ID, args.training_class)
    DatasetCatalog.register(f"robi_{args.training_class}_train",lambda: single_dataset_to_train)
    robi_meta=MetadataCatalog.get(f"robi_{args.training_class}_train").set(thing_classes=ROBI_CLASSES)

    # dataset_dicts = DatasetCatalog.get(f"robi_{args.training_class}_train")
    if args.test_dataset_loader:
        dataset_dicts = DatasetCatalog.get(f"robi_{args.training_class}_train")
        for d in random.sample(dataset_dicts, 3):
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=robi_meta, scale=0.5)
            out = visualizer.draw_dataset_dict(d)
            cv2.imshow('tab',out.get_image())
            # (this is necessary to avoid Python kernel form crashing)
            cv2.waitKey(0)

            # closing all open windows
            cv2.destroyAllWindows()

    #set up training
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.SOLVER.AMP.ENABLED = True 
    cfg.DATASETS.TRAIN = (f"robi_{args.training_class}_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size 
    cfg.SOLVER.BASE_LR = args.learning_rate
    cfg.SOLVER.MAX_ITER = args.num_iterations
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(ROBI_CLASSES)  # number of classes in the dataset


    #train
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

   

def loadPickle(pickle_file):
    dataset_dicts = pickle.load(pickle_file)
    if type(dataset_dicts) is not list:
        raise ValueError("Pickle file does not contain a list")
    elif type(dataset_dicts[0]) is not dict:
        raise ValueError("Pickle file does not contain a list of dictionaries")
    return dataset_dicts



def registerFromPickle(dataset_dicts):
    for class_name in ROBI_CLASSES:
        dataset_name = f"robi_{class_name}_train"
        print(f"registering dataset {dataset_name}")
        DatasetCatalog.register(dataset_name,)
        MetadataCatalog.get(dataset_name).set(thing_classes=ROBI_CLASSES)

def parser_setup(parser):
    parser.add_argument(
        "--training_class",
        type=str,
        default="Eye_bolt_synthetic",
        help="Class to train on"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=DataPath,
        help="Path to the ROBI dataset directory"
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=1000,
        help="Number of iterations to train"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training"
    )
    parser.add_argument(
        "--test_dataset_loader",
        action='store_true',  
        help="If specified, visualises dataset after loading"
    )
    parser.add_argument(
        "--evaluate",
        action='store_true',
        help="If specified, evaluate the model after training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.00025,
        help="Learning rate for the optimizer"
    )


if __name__ == "__main__":
    main()