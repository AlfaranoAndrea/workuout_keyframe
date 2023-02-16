import torch.nn.functional as F
import pytorch_lightning as pl
import torch
from loss import multiscaleEPE, realEPE
from flowNetSNet import FlowNetSNet

class FlowNetS(pl.LightningModule):
    def __init__(self, checkpoint=None):
        super().__init__()
        self.net= FlowNetSNet()
        
        
        if(checkpoint is not None):
             self.load_state_dict(checkpoint)
                
    def forward (self, x):
        return self.net(x)
      
        
    def training_step(self, batch, batch_idx):
        multiscale_weights = [0.005,0.01,0.02,0.08,0.32] 
            # how i weight loss on every level
            # i want to obtain the y resized to various level, in order to compute loss function:

        x,x1, y= batch    
        tensors= []
        for i in range(len(x1)):
            
            cat= torch.cat((x[i],x1[i]),1)
          #  cat= torch.reshape(cat, (1, *cat.size()))
            tensors.append(cat)
            
        x_train= torch.cat(tensors,0)
        output=self(x_train) 
        h, w = y.size()[-2:]  
    
        output = [F.interpolate(output[0], (h,w)), *output[1:]] # it upscale output image to make confrontable y with output      
        loss = multiscaleEPE(output, y, weights=multiscale_weights)
        self.log('train_loss', loss)
        return loss


    def test_step(self, batch, batch_idx):
        x,x1, y= batch    
        tensors= []
        for i in range(len(x1)):
            
            cat= torch.cat((x[i],x1[i]),1)
          #  cat= torch.reshape(cat, (1, *cat.size()))
            tensors.append(cat)
            
        x_train= torch.cat(tensors,0)
        output=self(x_train) 
        h, w = y.size()[-2:]  
    
        output = [F.interpolate(output[0], (h,w)), *output[1:]] # it upscale output image to make confrontable y with output      
        loss =  realEPE(output[0], y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5, betas=(0.9, 0.999))
        return optimizer
    
    def save(self, path= '/models'):
        torch.save(self.state_dict(), path)