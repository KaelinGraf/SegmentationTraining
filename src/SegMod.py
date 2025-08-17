import detectron2 as d2 
import torch,torchvision
import argparse
import os,numpy 
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
import random
import cv2
import pickle

MODEL_PATH ="/home/kaelin/Documents/BinPicking/SegmentationModel/output/model_0252999.pth"
ROBI_CLASSES = ["Chrome_screw_synthetic","DIN_connector_synthetic","DSub_connector_synthetic","Eye_bolt_synthetic","Gear_synthetic","Tube_fitting_synthetic","Zigzag_synthetic"]
CLASS_TO_ID = {cls: idx for idx, cls in enumerate(ROBI_CLASSES)}
DataPath = "/home/kaelin/Documents/BinPicking/SegmentationModel/Data/"
pickle_file = os.path.join("Data/robi_dataset.pkl")

class SegMod:
    _cfg = get_cfg()
    #_predictor = DefaultPredictor(_cfg)
    _parser = argparse.ArgumentParser(
        prog=""
    )
    def __init__(self):
        self.parser_setup()
        self._args = self._parser.parse_args()
        
        self._cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self._cfg.SOLVER.AMP.ENABLED = True 
        self._cfg.DATALOADER.NUM_WORKERS = 4
        self._cfg.MODEL.WEIGHTS = MODEL_PATH
        self._cfg.SOLVER.IMS_PER_BATCH = self._args.batch_size 
        self._cfg.SOLVER.BASE_LR = self._args.learning_rate
        self._cfg.SOLVER.MAX_ITER = self._args.num_iterations
        self._cfg.SOLVER.CHECKPOINT_PERIOD = 1000
        self._cfg.SOLVER.STEPS = tuple(range(50000, self._args.num_iterations, 50000))
        self._cfg.SOLVER.GAMMA = 0.8
        self._cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        self._cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(ROBI_CLASSES)  # number of classes in the dataset

    def inference(self,image_path):
        self._cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95
        self._predictor = DefaultPredictor(self._cfg)
        
        image = cv2.imread(image_path)
        if image is None:
            print("failed")
            exit()
        cv2.imshow('image',image)
        outputs = self._predictor(image)
        v = Visualizer(image[:, :, ::-1], scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("Result", out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return outputs


    def train(self):
        #train the model (either from scratch or from last checkpoints)
        all_datasets = self.loadPickle(pickle_file)
    
        training_datasets = []
        if self._args.training_class == "all":
            print("Combining all classes into a single dataset for training.")
            for class_name in ROBI_CLASSES:
                training_datasets.extend(all_datasets[class_name])
            
            training_dataset_name = "robi_train_all"
            DatasetCatalog.register(training_dataset_name, lambda: training_datasets)
            MetadataCatalog.get(training_dataset_name).set(thing_classes=ROBI_CLASSES)
        else:
            single_dataset_to_train = all_datasets[self._args.training_class]

            #registerROBISingleClassDataset(args.data_path, CLASS_TO_ID, args.training_class)
            DatasetCatalog.register(f"robi_{self._args.training_class}_train",lambda: single_dataset_to_train)
            robi_meta=MetadataCatalog.get(f"robi_{self._args.training_class}_train").set(thing_classes=ROBI_CLASSES)
            training_datasets.append[f"robi_{self._args.training_class}_train"]

        # dataset_dicts = DatasetCatalog.get(f"robi_{args.training_class}_train")
        if self._args.test_dataset_loader:
            dataset_dicts = DatasetCatalog.get(f"robi_{self._args.training_class}_train")
            for d in random.sample(dataset_dicts, 3):
                img = cv2.imread(d["file_name"])
                visualizer = Visualizer(img[:, :, ::-1], metadata=robi_meta, scale=0.5)
                out = visualizer.draw_dataset_dict(d)
                cv2.imshow('tab',out.get_image())
                # (this is necessary to avoid Python kernel form crashing)
                cv2.waitKey(0)

                # closing all open windows
                cv2.destroyAllWindows()

        self._cfg.DATASETS.TRAIN = (training_dataset_name)
        self._cfg.DATASETS.TEST = ()
        os.makedirs(self._cfg.OUTPUT_DIR, exist_ok=True)
        _trainer = DefaultTrainer(self._cfg) 
        _trainer.resume_or_load(resume=True)
        _trainer.train()
        
    def eval():
        #evaluate training progress on dataset
        pass

    def parser_setup(self):
        
        self._parser.add_argument(
            "--training_class",
            type=str,
            default="all",
            help="Class to train on"
        )
        self._parser.add_argument(
            "--data_path",
            type=str,
            default=DataPath,
            help="Path to the ROBI dataset directory"
        )
        self._parser.add_argument(
            "--num_iterations",
            type=int,
            default=1000000,
            help="Number of iterations to train"
        )
        self._parser.add_argument(
            "--batch_size",
            type=int,
            default=8,
            help="Batch size for training"
        )
        self._parser.add_argument(
            "--test_dataset_loader",
            action='store_true',  
            help="If specified, visualises dataset after loading"
        )
        self._parser.add_argument(
            "--evaluate",
            action='store_true',
            help="If specified, evaluate the model after training"
        )
        self._parser.add_argument(
            "--learning_rate",
            type=float,
            default=0.00015,
            help="Learning rate for the optimizer"
        )

    def loadPickle(self,pickle_file):
        with open(pickle_file,'rb') as f:
            dataset_dicts = pickle.load(f)
        if type(dataset_dicts) is not dict:
            raise ValueError("Pickle file does not contain a list of dictionaries")
        return dataset_dicts


            
        
if __name__ == "__main__":
    myModel = SegMod()

    im_path = "/home/kaelin/Downloads/bin2.png"
    print(im_path)
    myModel.inference(im_path)