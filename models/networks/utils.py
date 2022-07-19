from multiprocessing import reduction
import torch
import torch.nn.functional as F
import math
import torch
import torchvision.models.vgg as models
from torchvision import transforms
import numpy as np
from scipy.linalg import logm, norm
from torch.nn import functional as F
from PIL import Image
from models.networks.networks import euler2mat
from torchviz import make_dot
import pytorch3d.transforms as T
def init_variable(dim, n_init, device, mode='random', range=[0, 1], value=1):

    shape = (n_init, dim)
    var = torch.ones(shape, requires_grad=True,
                     device=device, dtype=torch.float)
    if mode == 'random':
        var.data = torch.rand(shape, device=device) * \
            (range[1]-range[0]) + range[0]
    elif mode == 'linspace':
        var.data = torch.linspace(
            range[0], range[1], steps=n_init, device=device).unsqueeze(-1)
    elif mode == 'constant':
        var.data = value*var.data
    else:
        raise NotImplementedError
    return var


def grid_sample(image, grid, mode='bilinear', padding_mode='constant', padding_value=1):
    image_out = F.grid_sample(image, grid, mode=mode, padding_mode='border', align_corners=True)
    if padding_mode == 'constant':
        out_of_bound = grid[:, :, :, 0] > 1
        out_of_bound += grid[:, :, :, 0] < -1
        out_of_bound += grid[:, :, :, 1] > 1
        out_of_bound += grid[:, :, :, 1] < -1
        out_of_bound = out_of_bound.unsqueeze(1).expand(image_out.shape)
        image_out[out_of_bound] = padding_value
    return image_out


def warping_grid(angle, transx, transy, scale, image_shape):
    cosz = torch.cos(angle)
    sinz = torch.sin(angle)
    affine_mat = torch.cat([cosz, -sinz, transx,
                            sinz,  cosz, transy], dim=1).view(image_shape[0], 2, 3)
    scale = scale.view(-1, 1, 1).expand(affine_mat.shape)
    return F.affine_grid(size=image_shape, theta=scale*affine_mat, align_corners=True)


def set_axis(ax):
    ax.clear()
    ax.xaxis.set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(axis='y')


def compute_angle_loss(angle, gt_angle):
    compute_loss = torch.nn.L1Loss()
    angle_loss = compute_loss(angle, gt_angle)
    angle_loss = angle_loss % 360
    if angle_loss >= 180 and angle_loss <= 360:
        angle_loss = 360 - angle_loss
    # print("angle", angle_loss)
    return angle_loss.detach().cpu().numpy()

# def compute_pose_error(angle, gt_angle):
#     R_1 = euler2mat(angle)
#     R_2 = euler2mat(gt_angle)
#     # print("matriox", R_1.shape, R_2.shape)
#     R_12 = torch.bmm(R_1, torch.transpose(R_2, 1, 2))
#     # print("R12", R_12.shape)
#     loss = (torch.einsum('bii->b', R_12)-1)/2
#     epsilon = 1e-6
#     loss = torch.clamp(loss, min=-1+epsilon, max=1-epsilon)
#     # print("loss 11", loss)
#     angle_loss = torch.acos(loss).mean()
#     # print("loss 2", angle_loss, angle_loss*180.0/np.pi)
#     # make_dot(angle_loss).render("angle_loss", format="png")
#     return angle_loss

def compute_pose_error(angle, gt_angle):
    R_1 = T.euler_angles_to_matrix(angle, "XYZ")
    R_2 = T.euler_angles_to_matrix(gt_angle, "XYZ")
    # print("matriox", R_1.shape, R_2.shape)
    angle = T.so3_relative_angle(R_1, R_2, True).mean()
    # print("angle", angle)
    
    return angle  

def euler_to_quaternion(input):
    roll = input[:, 0:1]
    pitch = input[:, 1:2]
    yaw = input[:, 2:3]
    qx = torch.sin(roll/2) * torch.cos(pitch/2) * torch.cos(yaw/2) - torch.cos(roll/2) * torch.sin(pitch/2) * torch.sin(yaw/2)
    qy = torch.cos(roll/2) * torch.sin(pitch/2) * torch.cos(yaw/2) + torch.sin(roll/2) * torch.cos(pitch/2) * torch.sin(yaw/2)
    qz = torch.cos(roll/2) * torch.cos(pitch/2) * torch.sin(yaw/2) - torch.sin(roll/2) * torch.sin(pitch/2) * torch.cos(yaw/2)
    qw = torch.cos(roll/2) * torch.cos(pitch/2) * torch.cos(yaw/2) + torch.sin(roll/2) * torch.sin(pitch/2) * torch.sin(yaw/2)

    return torch.stack([qx, qy, qz, qw], dim=1)

def compute_quat_loss(input, gt, symm=0, reduction='mean'):
    if symm==1:
        input[:,1] = 0
        gt[:,1] = 0
    compute_loss = torch.nn.MSELoss(reduction=reduction)
    return compute_loss(euler_to_quaternion(input), euler_to_quaternion(gt))
def compute_vgg_loss(image, gt_image):
    device = torch.device("cuda:0")
    vgg = models.vgg16(pretrained=True).to(device)
    normalization_mean = [0.485, 0.456, 0.406]
    normalization_std = [0.229, 0.224, 0.225]
    loader = transforms.Compose(
        [transforms.Normalize(mean=normalization_mean, std=normalization_std)])
    vgg_features_gt = vgg(loader(gt_image).to(device))
    vgg_features_image = vgg(loader(image).to(device))
    compute_loss = torch.nn.MSELoss()
    return compute_loss(vgg_features_gt, vgg_features_image)


def compute_l1_loss(image, gt_image, reduction='mean'):
    compute_loss = torch.nn.L1Loss(reduction=reduction)
    return compute_loss(image, gt_image)


def compute_l2_loss(image, gt_image, reduction='mean'):
    compute_loss = torch.nn.MSELoss(reduction=reduction)
    return compute_loss(image, gt_image)


def compute_pose_loss(R, R_gt, mode=1):
    if mode == 0:
        R, R_gt = map(np.matrix, [R, R_gt])
        _logRR, errest = logm(R.transpose()*R_gt, disp=False)
        loss  = norm(_logRR, 'fro') / np.sqrt(2)
        
    elif mode == 1:
        # print(R.shape, R_gt.shape)
        R, R_gt = map(np.matrix, [R, R_gt])
        # Do clipping to [-1,1].
        # For a few cases, (tr(R)-1)/2 can be a little bit less/greater than -1/1.
        logR_F = np.clip((np.trace(R*R_gt.transpose())-1.)/2., -1, 1)
        loss = np.arccos(logR_F)
    # print("poseloss", loss)
    return loss


def compute_RotMats(a, e, t, degree=True):
    # print("a e t", a.shape, e.shape, t.shape)
    batch = a.shape[0]
    Rz = np.zeros((batch, 3, 3), dtype=np.float32)
    Rx = np.zeros((batch, 3, 3), dtype=np.float32)
    Rz2 = np.zeros((batch, 3, 3), dtype=np.float32)
    Rz[:, 2, 2] = 1
    Rx[:, 0, 0] = 1
    Rz2[:, 2, 2] = 1
    #
    R = np.zeros((batch, 3, 3), dtype=np.float32)
    if degree:
        a = a * np.pi / 180.
        e = e * np.pi / 180.
        t = t * np.pi / 180.
    a = -a
    e = np.pi/2.+e
    t = -t
    #
    sin_a, cos_a = np.sin(a), np.cos(a)
    sin_e, cos_e = np.sin(e), np.cos(e)
    sin_t, cos_t = np.sin(t), np.cos(t)

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
    R[:] = np.einsum("nij,njk,nkl->nil", Rz2, Rx, Rz)

    # Return the original matrix without transpose!
    return R


def compute_dis_loss(d_outs, target):

    d_outs = [d_outs] if not isinstance(d_outs, list) else d_outs
    loss = 0

    for d_out in d_outs:

        targets = d_out.new_full(size=d_out.size(), fill_value=target)
        loss += F.binary_cross_entropy_with_logits(d_out, targets)
    return loss / len(d_outs)
