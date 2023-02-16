import torch
import torch.nn.functional as F

"""
adapted by: nvidia flownet 2.0 implementation and Clement Pinard flownet implementation
"""

def EPE(input_flow, target_flow, ):
    """
        return the expectation error between input and output flow, like in original paper
    """
    return torch.abs(target_flow-input_flow).mean()
    
def realEPE(output, target):
    """
        return the expectation error between input and output flow, performing the upsample of the computed flow
    """
    b, _, h, w = target.size()
    upsampled_output = F.interpolate(output, (h,w), mode='bilinear', align_corners=False) # used to resize the output
    return EPE(upsampled_output, target)


def multiscaleEPE(network_output, target_flow, weights=None):
    """
        input: network_output: batchtensor with all the computer flows at differents scale
        output: EPE between the computed flows and the groundtruth, weighted at different scales by a weight vector
    """
    def one_scale(output, target):

        b, _, h, w = output.size()                            # obtain batch size, hight and widht
       
        target_scaled = F.interpolate(target, (h, w), mode='area')    # down-size using nn.functional.interpolation
        return EPE(output, target_scaled)
    
    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    if weights is None:
        weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # as in original article
    assert(len(weights) == len(network_output))

    loss = 0
    for output, weight in zip(network_output, weights):
        loss += weight * one_scale(output, target_flow)
    return loss