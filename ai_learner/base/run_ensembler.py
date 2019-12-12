from ensembler import AugmentRotate, Inferer, ComplexAugment, AugmentFlip, BasicEnsembler, AAAREnsembler
import cv2
import torch
import torch.nn as nn
from os import listdir
import numpy as np
import pandas as pd
from os.path import join
import sys

CODE_PATH = '/storage/xview2/xview_gitlab/segmentation_models.pytorch'
sys.path.append(CODE_PATH)
CODE_PATH = '/storage/xview2/xview_gitlab/xview2_mmm/'
sys.path.append(CODE_PATH)
CODE_PATH = '/storage/xview2/xview_gitlab/ai_learner/'
sys.path.append(CODE_PATH)


models_dir = '/storage/xview2/project/models/'
models = {}
for model_name in listdir(models_dir):
    models[model_name] = torch.load(join(models_dir, model_name))
    break

folder = '/storage/xview2/project/xview_test/post/'
imgs = listdir(folder)
df = pd.DataFrame({
    'img_name': imgs,
    'dataset': ['xview_test'] * len(imgs),
})


inferer = Inferer(models=models,
                  predictions_dir='/storage/xview2/project/infer_test/',
                  df=df,
                  data_dir='/storage/xview2/project/',
                  input_img_folders=['pre', 'post'],
                  batch_size=14,
                  ttas=Inferer.BASE_AGUMENTS,
                  cuda_flag=True)
basic_ensembler = BasicEnsembler(inps_dir='/storage/xview2/project/infer_test/',
                                 outs_dir='/storage/xview2/project/ensemble_test/')
aaar_ensembler = AAAREnsembler(inps_dir='/storage/xview2/project/infer_test/',
                               outs_dir='/storage/xview2/project/ensemble_test/')


inferer.infer_all_models()
# basic_ensembler.predict_all()
aaar_ensembler.predict_all(single_process_flag=False)
