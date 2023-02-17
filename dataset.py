import pytorch_lightning as pl
import numpy as np
from torch.utils.data import  DataLoader
from torchvision import datasets, transforms

import torch 
class FlowDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=1, workers=1,selected_dataset="sintel" , camera='left'):
        super().__init__()

        self.download_dir = './dataset'
        self.batch_size = batch_size  
        self.workers = workers      
        self.camera = camera
        self.selected_dataset=selected_dataset
        
        self.input_transform=transforms.Compose([
                                             self.ArrayToTensor,
                                             transforms.Normalize(mean=[0,0,0], std=[255,255,255]), 
                                             transforms.Normalize(mean=[0.45,0.432,0.411], std=[1,1,1])
                                            ])
                                           
        self.flow_transforms= transforms.Compose([  
                                            self.FlowToTensor,
                                            transforms.Normalize(mean=[0,0],std=[20,20])
                                            ])
        # Defining transforms to be applied on the data
    
    def dataset_transformation(self, input1, input2,flow, valid_flow_mask):
        """
              train_transform takes data from torchVision Flyingchairs dataset and return a tensor ready for model training
              I choice to not performe data augmentation: dataset is so huge that overcome my computational capabilities :(,
              so i don't need to have more of theme :)
        """
        valid_flow_mask= None  #put none valid_flow_mask (only for compatibility with other dataset)
         #from PIL image to numpy
        input1 = np.asarray(input1)
        input2 = np.asarray(input2)
        flow= np.asarray(flow)
        input1= self.input_transform(input1)
        input2= self.input_transform(input2)       
        flow=self.flow_transforms(flow)     
        
        return input1, input2 , flow, valid_flow_mask
    
    def setup(self, stage):   
        if(self.selected_dataset == "FlyingThings3D"):
            self.train_data = datasets.FlyingThings3D(
                                        self.download_dir,
                                        split = "train", 
                                        transforms = self.dataset_transformation,
                                        pass_name = self.pass_name,
                                        camera=  self.camera                   
                                        )

            self.test_data = datasets.FlyingThings3D(
                                        self.download_dir,
                                        split = "val",
                                        transforms = self.dataset_transformation,
                                        pass_name = self.pass_name,
                                        camera=  self.camera               
                                        )
        
        elif(self.selected_dataset == "FlyingChairs"):
            self.train_data = datasets.FlyingChairs(
                                        self.download_dir,
                                        split = "train", 
                                        transforms = self.dataset_transformation
                                        )

            self.test_data = datasets.FlyingChairs( 
                                        self.download_dir,
                                        split = "val",
                                        transforms = self.dataset_transformation,
                                        )

        elif(self.selected_dataset=="Sintel"):   
            self.train_data = datasets.Sintel(
                                        self.download_dir,
                                        split = "train", 
                                        transforms = self.dataset_transformation,
                                        pass_name = "final"                  
                                        )

             self.test_data = None #datasets.Sintel(
#                                         self.download_dir,
#                                         split = "test",
#                                         transforms = self.dataset_transformation,
#                                         pass_name = "final"
#                                                         )
  
    def train_dataloader(self):
        return DataLoader(
                self.train_data, 
                batch_size = self.batch_size,
                num_workers = self.workers )

    def test_dataloader(self):
        return DataLoader(
                self.test_data,
                batch_size = self.batch_size,
                num_workers = self.workers )

    def ArrayToTensor(self, array):
        """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""
        assert(isinstance(array, np.ndarray))
        array2= np.float32(array) 
        array2 = np.transpose(array2, (2, 0, 1))
        tensor = torch.from_numpy(array2)
        return tensor.float()
    def FlowToTensor(self, array):
        """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""
        assert(isinstance(array, np.ndarray))
        array2= np.float32(array)   
        tensor = torch.from_numpy(array2)
        return tensor.float()
