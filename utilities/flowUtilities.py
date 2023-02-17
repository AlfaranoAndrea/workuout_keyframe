from dataset import  FlowDataModule
import flowiz as fz
import torch
import numpy as np
import os, os.path
import cv2
from tqdm import tqdm

def computeFlow(img1, img2, model, device):
    converter= FlowDataModule().input_transform  
    model.to(device)
    img1=converter(img1).to(device) 
    img2=converter(img2).to(device) 
    
    c, h, w = img1.shape  
    img1=torch.reshape(img1,(1, c, h,w))
    img2=torch.reshape(img2,(1, c, h,w))

    x = torch.cat([img1,img2],1)
    y=model(x)
    y_numpy=y[0][0].cpu().detach().numpy()
    y2=np.transpose (y_numpy,(1,2,0))
    flow=fz.convert_from_flow(y2) #i use flowviz to convert flow to image
    return flow


def flowVideo(path, model, device,step=4,temporary_path= "./temp/processedFloFrames/"):
    os.makedirs(temporary_path, exist_ok=True)
    files = os.listdir(path) 
    files.sort(key = lambda x: int(x[:-4]))
    
    image2 = cv2.imread(path+files[0])
    scale_percent = 40 # percent of original size
    width = int(image2.shape[1] * scale_percent / 100)
    height = int(image2.shape[0] * scale_percent / 100)
    image2 = cv2.resize(image2, (width, height), interpolation = cv2.INTER_AREA)
    for i in tqdm(range(1,len(files))[::step]):
        image1=image2
        image2 = cv2.imread(path+files[i])
        image2 = cv2.resize(image2, (width, height), interpolation = cv2.INTER_AREA)
        frame=computeFlow(image1,image2,model, device)
        cv2.imwrite(f"{temporary_path}{i}.jpg", frame)
