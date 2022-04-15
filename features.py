
import os
import random
import shutil
import time
import warnings
import sys
import numpy as np
import pdb
import logging
import matplotlib.pyplot as plt
import cv2
import configparser
from torchvision import datasets, models, transforms
import glob
import ast

from PIL import Image
from alexnet_fc7out_simplified_traffic_badnets_varying import alexnet, NormalizeByChannelMeanStd,AlexNet
from dataset import PoisonGenerationDataset
from dataset import LabeledDataset

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import scipy.io as sio

experimentID = "experiment0401"

data_root="Image"
txt_root="ImageNet_data_list"
seed="None"

num_iter=int("5000")

patched_root    = "patched_data"

clean_data_root = "Image"
poison_root = "patched_data"
patched_root    = "patched_data"
gpu         = int("0")
epochs      = int("100")
patch_size  = int("5")
eps         = int("32")
rand_loc    = "False"
trigger_id  = int("14")
num_poison  = int("800")
num_classes = int("43")
batch_size  = int("256")
logfile     = "logs/{}/finetune.log".format(experimentID, rand_loc, eps, patch_size, num_poison, trigger_id)
lr          = float("0.001")
momentum    = float("0.9")

target_wnid = "3"

source_wnid_list = "data/{}/source_wnid_list.txt".format(experimentID)
num_source = int("1")

saveDir_poison = "poison_data/" + experimentID + "/rand_loc_" +  str(rand_loc) + '/eps_' + str(eps) + \
                    '/patch_size_' + str(patch_size) + '/trigger_' + str(trigger_id) 
saveDir_patched = "patched_data/" + experimentID + "/rand_loc_" +  str(rand_loc) + '/eps_' + str(eps) + \
                    '/patch_size_' + str(patch_size) + '/trigger_' + str(trigger_id) 

trans_image = transforms.Compose([transforms.Resize((32, 32)),
                                      transforms.ToTensor(),
                                      ])

data_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        ])
    
    # SOURCE AND TARGET DATASETS
#target_filelist = "ImageNet_data_list/test/n03584254.txt"
#target_filelist = "ImageNet_data_list/finetune/3.txt"
target_filelist = "ImageNet_data_list/finetune/3.txt"
#target_filelist = "ImageNet_data_list/finetune/13.txt"
#target_filelist = "ImageNet_data_list/finetune/n07930864.txt"
#target_filelist = "ImageNet_data_list/finetune/5.txt"


    # Use source wnid list
if num_source==1:
    logging.info("Using single source for this experiment.")
else:
    logging.info("Using multiple source for this experiment.")

#with open("data/{}/multi_source_filelist.txt".format(experimentID),"w") as f1:
#   with open(source_wnid_list) as f2:
#       source_wnids = f2.readlines()
#       source_wnids = [s.strip() for s in source_wnids]

#       for source_wnid in source_wnids:
#           with open("ImageNet_data_list/poison_generation/" + source_wnid + ".txt", "r") as f2:
#               shutil.copyfileobj(f2, f1)

dataset_target = PoisonGenerationDataset(data_root + "/train", target_filelist, trans_image)
train_loader_target = torch.utils.data.DataLoader(dataset_target,
                                                    #batch_size=1083,
                                                    #batch_size=1210,
                                                    #batch_size=800,
                                                    #batch_size=1360,
                                                    #batch_size=900,
                                                    #batch_size=1760,
                                                    batch_size=len(dataset_target),
                                                    shuffle=True,
                                                    num_workers=0,
                                                    pin_memory=True)

for index in range(43):
    source_filelist = "ImageNet_data_list/finetune/" + str(index) + ".txt"


    dataset_source = PoisonGenerationDataset(data_root + "/train", source_filelist, trans_image)

    # SOURCE AND TARGET DATALOADERS

    train_loader_source = torch.utils.data.DataLoader(dataset_source,
                                                     batch_size=len(dataset_source),
                                                     shuffle=False,
                                                     num_workers=0,
                                                     pin_memory=True)



    saveDir = poison_root + "/" + experimentID + "/rand_loc_" +  str(rand_loc) + "/eps_" + str(eps) + \
                    "/patch_size_" + str(patch_size) + "/trigger_" + str(trigger_id)
    filelist = sorted(glob.glob(saveDir + "/*"))

    dataset_poison = LabeledDataset(saveDir,
                                "data/{}/poison_filelist1.txt".format(experimentID), data_transforms)


#experimentID = "experiment0401"
    saveDir = patched_root + "/" + experimentID + "/rand_loc_" +  str(rand_loc) + "/eps_" + str(eps) + \
                    "/patch_size_" + str(patch_size) + "/trigger_" + str(trigger_id)
    filelist = sorted(glob.glob(saveDir + "/*"))
    random.shuffle(filelist)

    dataset_patched = LabeledDataset(saveDir,
                                "data/{}/patched_filelist.txt".format(experimentID), data_transforms)



#poison_filelist = "data/experiment0011/poison_filelist.txt"

#train_loader_poison = torch.utils.data.DataLoader(dataset_poison,
#                                                     batch_size=len(dataset_poison),
#                                                     shuffle=False,
#                                                     num_workers=0,
#                                                     pin_memory=True)

    train_loader_patched = torch.utils.data.DataLoader(dataset_poison,
                                                      batch_size=len(dataset_poison),
                                                      shuffle=False,
                                                      num_workers=0,
                                                      pin_memory=True)


    logging.info("Number of index:{}".format(index))
    logging.info("Number of target images:{}".format(len(dataset_target)))
    logging.info("Number of source images:{}".format(len(dataset_source)))
    logging.info("Number of poison images:{}".format(len(dataset_poison)))
    logging.info("Number of patched images:{}".format(len(dataset_patched)))

    model1 = alexnet(pretrained=True)
    model1.to("cpu")
    model1.eval()

    # USE ITERATORS ON DATALOADERS TO HAVE DISTINCT PAIRING EACH TIME
    iter_target = iter(train_loader_target)
    iter_source = iter(train_loader_source)


    iter_patched = iter(train_loader_patched)

#num_poisoned = 0
#for i in range(len(train_loader_target)):

        # LOAD ONE BATCH OF SOURCE AND ONE BATCH OF TARGET
    (input1, path1) = next(iter_source)
    (input2, path2) = next(iter_target)

    (input6, path6) = next(iter_patched)

    x1 = input1
    feature11, feature12 = model1.forward(x1)
    feature12 = feature12.detach().numpy()
    feature11 = feature11.detach().numpy()
    
    x2 = input2
    feature21, feature22 = model1.forward(x2)
    feature22 = feature22.detach().numpy()
    feature21 = feature21.detach().numpy()

    #checkpointDir =  "finetuned_models/" + experimentID + "/rand_loc_" +  str(rand_loc) + "/eps_" + str(eps) + \
     #           "/patch_size_" + str(patch_size) + "/num_poison_" + str(num_poison) + "/trigger_" + str(trigger_id) + "/badnets"

    x6 = input6
    feature61, feature62 = model1.forward(x6)
    feature62 = feature62.detach().numpy()
    #feature61 = feature61.detach().numpy()

    saveDir_features = "untitled_folder/" + experimentID + "/rand_loc_" +  str(rand_loc) + '/eps_' + str(eps) + \
                    '/patch_size_' + str(patch_size) + '/trigger_' + str(trigger_id) + "/num_poison_" + str(num_poison) + '/badnets' + '/varying' + '/feats'

                    
    if not os.path.exists(saveDir_features):
        os.makedirs(saveDir_features)


#sio.savemat(saveDir_features + '/features_2class_poison.mat', {'features_source': feature12,'features_target': feature22})
#sio.savemat(saveDir_features + '/features_2class_target_patched.mat.mat', {'features_target_patched': feature52})
#sio.savemat(saveDir_features + '/features_2class_poison.mat', {'features_poison': feature32})
#sio.savemat(saveDir_features + '/features_2class_source_patched.mat', {'features_source_patched': feature42})
    sio.savemat(saveDir_features + '/features_2class_source' + str(index) + '.mat', {'features_source': feature12})
###sio.savemat(saveDir_features + '/features_2class_target.mat', {'features_target': feature22})
    sio.savemat(saveDir_features + '/features_2class_target.mat', {'features_target': feature22})
    sio.savemat(saveDir_features + '/features_2class_patched.mat', {'features_patched': feature62})

#sio.savemat('features_2class_target_patched.mat', {'features_target_patched': feature52})
#sio.savemat('features_2class_poison.mat', {'features_poison': feature32})
#sio.savemat('features_2class_source_patched.mat', {'features_source_patched': feature42})
#sio.savemat('features_2class_source.mat', {'features_source': feature12})
#sio.savemat('features_2class_target.mat', {'features_target': feature22})
                                                        
##sio.savemat(saveDir_features + '/conv_features_2class_source' + str(index) + '.mat', {'conv_features_source': feature12})


##sio.savemat(saveDir_features + '/output_2class_source' + str(index) + '.mat', {'output_source': feature11})

##sio.savemat(saveDir_features + '/output_2class_patched' + str(index) + '.mat', {'output_patched': feature61})

##sio.savemat(saveDir_features + '/conv_features_2class_patched' + str(index) + '.mat', {'conv_features_patched': feature62})


#x11 = x1.view(x1.size(0),3*32*32)
#x22 = x2.view(x2.size(0),3*32*32)
#x33 = x3.view(x3.size(0),3*32*32)
#x66 = x6.view(x6.size(0),3*32*32)

#x11 = x11.detach().numpy()
#x22 = x22.detach().numpy()
#x33 = x33.detach().numpy()
#x66 = x66.detach().numpy()

#x1 = x1.detach().numpy()

#x6 = x6.detach().numpy()

#sio.savemat(saveDir_features + '/inputs_2class_target.mat', {'inputs_target': x22})
##sio.savemat(saveDir_features + '/inputs_2class_source' + str(index) + '.mat', {'inputs_source': x11})
#sio.savemat(saveDir_features + '/inputs_2class_source42.mat', {'inputs_source': x1})
#sio.savemat(saveDir_features + '/inputs_2class_poison.mat', {'inputs_poison': x33})

##sio.savemat(saveDir_features + '/inputs_2class_patched' + str(index) + '.mat', {'inputs_patched': x66})