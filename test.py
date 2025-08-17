import detectron2 as d2
import torch
from detectron2.utils.logger import setup_logger
import ROBI_Toolkit
import src.robi_loader_detectron as robi_loader
import src.train_maskrcnn as train_maskrcnn


train_maskrcnn.train_maskrcnn()



