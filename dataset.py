class FlowDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, workers,dataset="sintel" , camera='left'):
        super().__init__()

        self.download_dir = './root'    # Directory to store FlyingChairs Data     
        self.batch_size =  batch_size   # Defining batch size of our data 
        self.workers =    workers      
        self.camera = camera
        self.dataset=dataset
        
        self.input_transform=transforms.Compose( [flow_transforms.ArrayToTensor(),
                                           transforms.Normalize(mean=[0,0,0], std=[255,255,255]), 
                                            transforms.Normalize(mean=[0.45,0.432,0.411], std=[1,1,1])])
                                           
        self.flow_transforms= transforms.Compose([  flow_transforms.FlowToTensor(),
                                            transforms.Normalize(mean=[0,0],std=[div_flow,div_flow])])
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
    
    def setup(self, stage=None):
        
        if(self.dataset == "FlyingThings3D"):
                self.train_data = datasets.FlyingThings3D(self.download_dir,
                              split = "train", 
                              transforms = self.dataset_transformation,
                              pass_name = self.pass_name,
                              camera=  self.camera                   )

                self.test_data = datasets.FlyingThings3D(self.download_dir,
                                        split = "val",
                                        transforms = self.dataset_transformation,
                                        pass_name = self.pass_name,
                                        camera=  self.camera               
                                                        )
        if(self.dataset == "FlyingChairs"):
            self.train_data = datasets.FlyingChairs(self.download_dir,
                              split = "train", 
                              transforms = self.dataset_transformation)

            self.test_data = datasets.FlyingChairs(self.download_dir,
                                        split = "val",
                                        transforms = self.dataset_transformation,
                                              )
        if(self.dataset=="Sintel"):   
            self.train_data = datasets.Sintel(self.download_dir,
                              split = "train", 
                              transforms = self.dataset_transformation,
                              pass_name = "clean"                  )

            self.test_data = datasets.Sintel(self.download_dir,
                                        split = "train",
                                        transforms = self.dataset_transformation,
                                        pass_name = "final"
                                                        )
  
    def train_dataloader(self):
        
          # Generating train_dataloader
        return DataLoader(self.train_data, 
                          batch_size = self.batch_size,
                          num_workers = self.workers )

    def test_dataloader(self):
        
        # Generating test_dataloader
        return DataLoader(self.test_data,
                          batch_size = self.batch_size,
                          num_workers = self.workers )
