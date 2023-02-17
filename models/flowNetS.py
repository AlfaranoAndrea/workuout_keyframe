import torch.nn.functional as F
import pytorch_lightning as pl
import torch
from loss import multiscaleEPE, realEPE
from models.flowNetSNet import FlowNetSNet

class FlowNetS(pl.LightningModule):
    def __init__(self, checkpoint=None, lr=1e-4):
        super().__init__()
        self.net= FlowNetSNet()
        self.lr=lr
        
        
        if(checkpoint is not None):
            self.net.load_state_dict(checkpoint)
                
    def forward (self, x):
        return self.net(x)
      
        
    def training_step(self, batch, batch_idx):
        multiscale_weights = [0.005,0.01,0.02,0.08,0.32] 
            # how i weight loss on every level
            # i want to obtain the y resized to various level, in order to compute loss function:

        x,x1, y= batch    
        x_train= torch.cat((x,x1),1)
        output=self(x_train) 
        h, w = y.size()[-2:]  
    
        output = [F.interpolate(output[0], (h,w)), *output[1:]] # it upscale output image to make confrontable y with output      
        loss = multiscaleEPE(output, y, weights=multiscale_weights)
        self.log('train_loss', loss)
        return loss


    def test_step(self, batch, batch_idx):
        x,x1, y= batch    
        x_train= torch.cat((x,x1),1)
        output=self(x_train) 
        h, w = y.size()[-2:]  
    
       # output = [F.interpolate(output[0], (h,w)), *output[1:]] # it upscale output image to make confrontable y with output      
        loss =  realEPE(output[0], y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
    
    def save(self, path= '/models'):
        torch.save(self.net.state_dict(), path)
