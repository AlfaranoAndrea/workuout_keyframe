from torchvision.models.optical_flow import raft_small, raft_large
import pytorch_lightning as pl

class RAFT(pl.LightningModule):
    def __init__(self, version="large"):
        super().__init__()
        if version=="large":
            self.net=raft_large(pretrained=True)
        else:
            self.net=raft_small(pretrained=True)

    def forward(self, img1,img2):
        return self.net(img1,img2)

    def load(self, device):
        self.net.to(device)
        
    def training_step(self, batch, batch_idx):
        pass

    def training_step(self, batch, batch_idx):
        pass


    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass