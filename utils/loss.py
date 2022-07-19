import math
import torch
import torchvision.models.vgg as models
from torchvision import transforms
import numpy as np
from scipy.linalg import logm, norm
import torch.nn as nn
def trans_error(input, targets):
    targets = targets[:, 0:1] * targets[:, 1:3] 
    input = input[:, 0:1] * input[:, 1:3] 
    shift = torch.norm(targets-input, dim=1) * 100
    return shift

def compute_RotMats(rot, degree=False):
    # print("a e t", a.shape, e.shape, t.shape)
    a, e, t = rot[:, 0], rot[:, 1], rot[:, 2]
    batch = rot.shape[0]
    Rz = torch.zeros((batch, 3, 3), dtype=torch.float32)
    Rx = torch.zeros((batch, 3, 3), dtype=torch.float32)
    Rz2 = torch.zeros((batch, 3, 3), dtype=torch.float32)
    Rz[:, 2, 2] = 1
    Rx[:, 0, 0] = 1
    Rz2[:, 2, 2] = 1
    #
    R = torch.zeros((batch, 3, 3), dtype=torch.float32)
    if degree:
        a = a * np.pi / 180.
        e = e * np.pi / 180.
        t = t * np.pi / 180.
    a = -a
    e = np.pi/2.+e
    t = -t
    #
    sin_a, cos_a = torch.sin(a), torch.cos(a)
    sin_e, cos_e = torch.sin(e), torch.cos(e)
    sin_t, cos_t = torch.sin(t), torch.cos(t)

    # ===========================
    #   rotation matrix
    # ===========================
    """
    # [Transposed]
    Rz = np.matrix( [[  cos(a), sin(a),       0 ],     # model rotate by a
                    [ -sin(a), cos(a),       0 ],
                    [      0,       0,       1 ]] )
    # [Transposed]
    Rx = np.matrix( [[      1,       0,       0 ],    # model rotate by e
                    [      0,   cos(e), sin(e) ],
                    [      0,  -sin(e), cos(e) ]] )
    # [Transposed]
    Rz2= np.matrix( [[ cos(t),   sin(t),      0 ],     # camera rotate by t (in-plane rotation)
                    [-sin(t),   cos(t),      0 ],
                    [      0,        0,      1 ]] )
    R = Rz2*Rx*Rz
    """

    # Original matrix (None-transposed.)
    # No need to set back to zero?
    Rz[:, 0, 0],  Rz[:, 0, 1] = cos_a, -sin_a
    Rz[:, 1, 0],  Rz[:, 1, 1] = sin_a,  cos_a
    #
    Rx[:, 1, 1],  Rx[:, 1, 2] = cos_e, -sin_e
    Rx[:, 2, 1],  Rx[:, 2, 2] = sin_e,  cos_e
    #
    Rz2[:, 0, 0], Rz2[:, 0, 1] = cos_t, -sin_t
    Rz2[:, 1, 0], Rz2[:, 1, 1] = sin_t,  cos_t
    # R = Rz2*Rx*Rz
    R[:] = torch.einsum("nij,njk,nkl->nil", Rz2, Rx, Rz)

    # Return the original matrix without transpose!
    return R
     
def rot_error(pose, pose_gt, reduction=False):
    R_pds = compute_RotMats(pose)
    R_gts = compute_RotMats(pose_gt)
    errors = []
    for i in range(R_gts.shape[0]): 
        R = R_pds[i] @  torch.transpose(R_gts[i], 0, 1)
        error = (torch.trace(R) - 1)/2
        error = torch.clamp(error, min=-1, max=1)
        error = torch.arccos(error) * 180/np.pi
        errors.append(error)
    if reduction:
        errors = sum(errors) / pose.shape[0]
    else:
        errors = torch.tensor(errors)
    return errors



def euler_to_quaternion(input):
        roll = input[:, 0:1]
        pitch = input[:, 1:2]
        yaw = input[:, 2:3]
        qx = torch.sin(roll/2) * torch.cos(pitch/2) * torch.cos(yaw/2) - torch.cos(roll/2) * torch.sin(pitch/2) * torch.sin(yaw/2)
        qy = torch.cos(roll/2) * torch.sin(pitch/2) * torch.cos(yaw/2) + torch.sin(roll/2) * torch.cos(pitch/2) * torch.sin(yaw/2)
        qz = torch.cos(roll/2) * torch.cos(pitch/2) * torch.sin(yaw/2) - torch.sin(roll/2) * torch.sin(pitch/2) * torch.cos(yaw/2)
        qw = torch.cos(roll/2) * torch.cos(pitch/2) * torch.cos(yaw/2) + torch.sin(roll/2) * torch.sin(pitch/2) * torch.sin(yaw/2)

        return torch.stack([qx, qy, qz, qw], dim=1)
    
class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, args):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.weight_dict = {
            'rot_loss': 10,
            'trans_loss': 5,
            'latent_loss': 1,
            'rot_distance': 0,
            'trans_distance': 0
        }
        self.loss_map = {
            'rot_loss': self.compute_quat_loss,
            'trans_loss': self.compute_l2_loss,
            'latent_loss': self.compute_l2_loss,
            'rot_distance': self.compute_rot_loss,
            'trans_distance': self.compute_trans_loss,
        }
        self.reduction= 'mean'
        self.category = args.dataset
        self.categories = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
        if self.category in ['bottle', 'can', 'bowl']:
            self.symm = True
        elif self.category in ['laptop', 'mug', 'camera']:
            self.symm = False
        self.l2_loss = torch.nn.MSELoss(reduction=self.reduction)
        self.l1_loss = torch.nn.L1Loss(reduction=self.reduction)
        self.vgg = models.vgg16(pretrained=True)
        

    def forward(self, outputs, targets, update_losses):
        losses = {}
        for loss in update_losses:
            losses.update(self.get_loss(loss, outputs[loss], targets[loss]))

        return losses

    def get_loss(self, loss, outputs, targets):

        assert loss in self.loss_map, f'do you really want to compute {loss} loss?'
        return self.loss_map[loss](outputs, targets, name=loss)
    
    def compute_vgg_loss(self, input, targets, name):
        
        normalization_mean = [0.485, 0.456, 0.406]
        normalization_std = [0.229, 0.224, 0.225]
        loader = transforms.Compose(
            [transforms.Normalize(mean=normalization_mean, std=normalization_std)])
        vgg_features_gt = self.vgg(loader(targets))
        vgg_features_image = self.vgg(loader(input))
        loss = self.l2_loss(vgg_features_gt, vgg_features_image)
        losses = {name: loss}

        return losses


    def compute_l1_loss(self, input, targets, name):
        loss = self.l1_loss(input, targets)
        losses = {name: loss}
        return losses
    
    def compute_quat_loss(self, input, gt, name):
        input = input * np.pi
        gt = gt * np.pi
        if self.symm == True:
            input[:, 1] = 0
            gt[:, 1] = 0
        loss = self.l2_loss(euler_to_quaternion(input), euler_to_quaternion(gt))
        losses = {name: loss}
        return losses

    def compute_l2_loss(self, input, targets, name):
        loss = self.l2_loss(input, targets)
        losses = {name: loss}
        return losses


    def compute_rot_loss(self, input, targets, name):
        input = input * np.pi
        targets = targets * np.pi
        if self.symm == True:
            input[:, 1] = 0
            targets[:, 1] = 0
        loss = rot_error(input, targets)
        losses = {name: loss}
        return losses
    
    def compute_trans_loss(self, T, T_gt, name):
        loss = trans_error(T, T_gt)
        losses = {name: loss}
        return losses
    
    

    