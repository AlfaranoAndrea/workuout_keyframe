from blocks import convolution_unit, deconvolution_unit,predict_flow,crop_like
import torch.nn as nn
import torch

class FlowNetSNet(nn.Module):
    def __init__(self, fn):
        super().__init__()
        #ENCODER PART OF THE NETWORK
       
        self.conv1   = convolution_unit( 6,   64, kernel_size=7, stride=2)
        self.conv2   = convolution_unit(  64,  128, kernel_size=5, stride=2)
        self.conv3   = convolution_unit(128,  256, kernel_size=5, stride=2)
        self.conv3_1 = convolution_unit( 256,  256)
        self.conv4   = convolution_unit( 256,  512, stride=2)
        self.conv4_1 = convolution_unit( 512,  512)
        self.conv5   = convolution_unit( 512,  512, stride=2)
        self.conv5_1 = convolution_unit( 512,  512)
        self.conv6   = convolution_unit( 512, 1024, stride=2)
        self.conv6_1 = convolution_unit(1024, 1024)      
        
        #REFINEMENT PART OF THE NETWORK
        self.deconv5 = deconvolution_unit(1024,512)
        self.deconv4 = deconvolution_unit(1026,256)
        self.deconv3 = deconvolution_unit(770,128)
        self.deconv2 = deconvolution_unit(386,64)

        self.predict_flow6 = predict_flow(1024) 
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

    def forward(self, x):
        # ENCODER
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))
        
        # REFINEMENT
        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)
        out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)

        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        flow5       = self.predict_flow5(concat5)
        flow5_up    = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
        out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)

        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        flow4       = self.predict_flow4(concat4)
        flow4_up    = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)

        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        flow3       = self.predict_flow3(concat3)
        flow3_up    = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        flow2 = self.predict_flow2(concat2)
        
        return flow2,flow3,flow4,flow5,flow6