from scipy.sparse.linalg.dsolve.linsolve import factorized
from options.test_options import TestOptions
import torch
import numpy as np
import tqdm
import pickle
import cv2
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as scipy_rot
import gym
from gym import spaces
from models.networks.utils import grid_sample, warping_grid, init_variable
from models.networks.losses import PerceptualLoss
from options.test_options import TestOptions
from data import create_dataset
from collections import OrderedDict
from torchvision import transforms
import PIL
import random
from models import create_model
from models.networks.utils import grid_sample, warping_grid, init_variable
import tqdm
import torch.optim as optim
import torch.nn.functional as F
from utils import loss
from utils.utils import is_between
from models.networks.utils import grid_sample, warping_grid, init_variable
from models.networks.losses import PerceptualLoss

class nocs_gym():
    def __init__(self, args, parser, criterion):
        super(nocs_gym, self).__init__()
        random.seed(args.seed)
        np.random.seed(args.seed)
        opt = TestOptions().parse(parser)  # get test options
        opt.num_threads = 1
        opt.serial_batches = True
        opt.no_flip = True
        opt.n_views = 2592
        print('------------- Creating Dataset ----------------')
        self.opt = opt
        opt.category = args.dataset
        opt.exp_name = args.dataset
        self.category = args.dataset
        self.categories = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']

        if self.category in ['bottle', 'can', 'bowl']:
            self.symm = True
        elif self.category in ['laptop', 'mug', 'camera']:
            self.symm = False
        self.dataset = create_dataset(opt).dataset
        self.n_samples = len(self.dataset)
        self.limits = {
            'ax': [0, 1 / 2],
            'ay': [-1 / 2, 1 / 2],
            'az': [-50/180.0, 50/180.0],
            's': [0.9, 1.2],
            'tx': [-0.2, 0.2],
            'ty': [-0.2, 0.2],
            'z': [-3, 3]
        }
        print('-------------- Creating Model -----------------')
        self.current_step = 0
        self.max_step = args.max_step
        model = create_model(opt)
        model.setup(opt)
        model.eval()
        self.model = model
        self.criterion = criterion
        self.action_space = torch.zeros(22)
        self.use_encoder = False
        self.loss_dict = [
                            'rot_loss',
                            'trans_loss',
                            'latent_loss'
                        ]
        self.eval_dict = ['rot_distance',
                        'trans_distance']
        self.step_dict = {"obs_image": None,
                          "action": None}
        
    def warp_image(self,
                   img,
                   az,
                   s=torch.tensor([[1]]),
                   tx=torch.tensor([[0]]),
                   ty=torch.tensor([[0]]), grad=False):
        az = az * np.pi
        grid = warping_grid(az, tx, ty, s, img.shape)
        img = grid_sample(img, grid)
        if grad == False:
            img = img.detach()
        return img
    
    def random_state(self, num, aug=False, bright=False):
        for i in range(num):
            choose = False
            while choose is False:
                random_index = int(np.random.random() * self.n_samples)
                data = self.dataset[random_index]
                if is_between(180, data['B_pose'][1], 360):
                    data['B_pose'][1] -= 360
                data['B_pose'] = data['B_pose'] / 180.0
               
                if  is_between(self.limits['ax'][0], data['B_pose'][0], self.limits['ax'][1]):
                    if self.symm == True:
                        choose = True
                    if self.symm == False and is_between(
                            self.limits['ay'][0], data['B_pose'][1], self.limits['ay'][1]):
                        choose = True
            gt_image, self.gt_info.data[i, :3] = data['B'].unsqueeze(
                0).to(self.model.device), torch.from_numpy(data['B_pose']).to(self.model.device)
            self.gt_info.data[i, 2:6] = torch.tensor([
                np.random.uniform(low=self.limits['az'][0],
                                  high=self.limits['az'][1]),
                np.random.uniform(low=self.limits['s'][0],
                                  high=self.limits['s'][1]),
                np.random.uniform(low=self.limits['tx'][0],
                                  high=self.limits['tx'][1]),
                np.random.uniform(low=self.limits['ty'][0],
                                  high=self.limits['ty'][1])
            ])
            gt_image = self.warp_image(gt_image,
                                       self.gt_info[i:i+1, 2:3],
                                       s=self.gt_info[i:i+1, 3:4],
                                       tx=self.gt_info[i:i+1, 4:5],
                                       ty=self.gt_info[i:i+1, 5:6])
            self.gt_images = torch.cat([self.gt_images, gt_image], dim=0)
        

    def reset(self, real=None, aug=False, batch_size=1):
        self.current_step = 0
        choose = batch_size
        self.batch_size = batch_size
        self.gt_info = torch.zeros(self.batch_size, 22)
        self.gt_images = torch.tensor(())
        if real is not None:
            real = real[:, [2, 1, 0], :, :]
            self.gt_images = real.clone()
        else:
            self.random_state(self.batch_size)
        self.ax = init_variable(dim=1,
                                n_init=batch_size,
                                device=self.model.device,
                                mode='constant',
                                value=1.0/4.0)

        self.ay = init_variable(dim=1,
                                n_init=batch_size,
                                device=self.model.device,
                                mode='constant',
                                value=0)

        self.az = init_variable(dim=1,
                                n_init=batch_size,
                                device=self.model.device,
                                mode='constant',
                                value=0)

        self.s = init_variable(dim=1,
                               n_init=batch_size,
                               device=self.model.device,
                               mode='constant',
                               value=np.random.uniform(
                                   low=self.limits['s'][0],
                                   high=self.limits['s'][1]))
        self.tx = init_variable(dim=1,
                                n_init=batch_size,
                                device=self.model.device,
                                mode='constant',
                                value=np.random.uniform(
                                    low=self.limits['tx'][0],
                                    high=self.limits['tx'][1]))
        self.ty = init_variable(dim=1,
                                n_init=batch_size,
                                device=self.model.device,
                                mode='constant',
                                value=np.random.uniform(
                                    low=self.limits['ty'][0],
                                    high=self.limits['ty'][1]))
        self.z = init_variable(dim=16,
                               n_init=batch_size,
                               device=self.model.device,
                               mode='constant',
                               value=0)
        latent = self.model.netE(
            F.interpolate(self.gt_images,
                          size=self.model.opt.crop_size,
                          mode='nearest'))
        if self.model.opt.use_VAE:
            mu, logvar = latent[:, :self.model.opt.
                                z_dim], latent[:, self.model.opt.z_dim:]
            std = logvar.mul(0.5).exp_()
            eps = self.model.z_dist.sample((1, ))
            self.gt_info.data[:, 6:] = eps.mul(std).add_(mu)
        if self.use_encoder:
            self.z.data = self.gt_info.data[:, 6:]

        angle = 180 * torch.cat(
            [self.ax, self.ay, torch.zeros_like(self.ay)], dim=1)
        fake_B = self.model.netG(self.z, angle)
        fake_B = self.warp_image(fake_B, self.az, self.s, self.tx, self.ty)
        state = torch.cat([self.gt_images.unsqueeze(1),
                           fake_B.unsqueeze(1)],
                          dim=1).detach()
        return state

    def sample_action(self, random=False):
        if random == True:
            factor = np.random.random()
        else:
            factor = 1.0
        action = factor * (self.gt_info - torch.cat(
            [self.ax, self.ay, self.az, self.s, self.tx, self.ty, self.z],
            dim=1))
        action = torch.clamp(action, min=-1, max=1)
        return action
    
    def evaluate(self):
        targets = {
            'rot_distance': self.gt_info[:, 0:3],
            'trans_distance': self.gt_info[:, 3:6],
        }
        rot = torch.cat([self.ax, self.ay, self.az], dim=1)
        trans = torch.cat([self.s, self.tx, self.ty], dim=1)
        outputs = {
            'rot_distance': rot,
            'trans_distance': trans,
        }
        loss_dict = self.criterion(outputs, targets, self.eval_dict)
        return loss_dict
        
    def calc_loss(self, action, eval=False):
        action_gt = self.sample_action()
        targets = {
            'rot_loss': action_gt[:, 0:3],
            'trans_loss': action_gt[:, 3:6],
            'latent_loss': action_gt[:, 6:22]
        }
        outputs = {
            'rot_loss': action[:, 0:3],
            'trans_loss': action[:, 3:6],
            'latent_loss': action[:, 6:22]
        }
        if eval == False:
            loss_dict = self.criterion(outputs, targets, self.loss_dict)
            weight_dict = self.criterion.weight_dict

            self.losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys()
                    if k in weight_dict)
            self.log_losses = {
                    "train/" + k: (loss_dict[k] * weight_dict[k])
                    for k in loss_dict.keys() if k in weight_dict
                }
            self.log_losses["train/loss"] = self.losses

        else:
            loss_dict = self.criterion(outputs, targets, self.loss_dict)
            weight_dict = self.criterion.weight_dict
        
            self.losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys()
                    if k in weight_dict)
            self.log_losses = {
                    "test/" + k: (loss_dict[k] * weight_dict[k])
                    for k in loss_dict.keys() if k in weight_dict
                }
            self.log_losses["test/loss"] = self.losses

    def step(self, action, eval=False):
        decay_factor = 1
        self.current_step += 1
        self.calc_loss(action, eval)
        self.ax.data += action[:, 0:1] *1/2*decay_factor
        self.ay.data += action[:, 1:2] *1/2* decay_factor
        self.az.data += action[:, 2:3] *1/2* decay_factor
        self.s.data += action[:, 3:4] * decay_factor
        self.tx.data += action[:, 4:5] * decay_factor
        self.ty.data += action[:, 5:6] * decay_factor
        self.z.data += action[:, 6:22] * decay_factor
        self.ax.data = torch.clamp(self.ax.data,
                                   min=self.limits['ax'][0],
                                   max=self.limits['ax'][1])
        self.ay.data = torch.clamp(self.ay.data,
                                   min=self.limits['ay'][0],
                                   max=self.limits['ay'][1])
        self.az.data = torch.clamp(self.az.data,
                                   min=self.limits['az'][0],
                                   max=self.limits['az'][1])
        self.s.data = torch.clamp(self.s.data,
                                  min=self.limits['s'][0],
                                  max=self.limits['s'][1])
        self.tx.data = torch.clamp(self.tx.data,
                                   min=self.limits['tx'][0],
                                   max=self.limits['tx'][1])
        self.ty.data = torch.clamp(self.ty.data,
                                   min=self.limits['ty'][0],
                                   max=self.limits['ty'][1])
        self.z.data = torch.clamp(self.z.data,
                                  min=self.limits['z'][0],
                                  max=self.limits['z'][1])

        done = False
        angle = 180 * \
            torch.cat([self.ax, self.ay,
                       torch.zeros_like(self.ay)], dim=1)
        fake_B = self.model.netG(self.z, angle)
        fake_B = self.warp_image(fake_B, self.az, self.s, self.tx,
                                 self.ty)
        state = torch.cat([self.gt_images.unsqueeze(1),
                           fake_B.unsqueeze(1)],
                          dim=1).detach()
        
        return state
        
    def optimize(self, iter=10):
        real_B = self.gt_images
        variable_dict = [
            {
                'params': self.z,
                'lr': 3e-1
            },
            {
                'params': self.ax,
                'lr': 1e-2
            },
            {
                'params': self.ay,
                'lr': 3e-2
            },
            {
                'params': self.az,
                'lr': 1e-2
            },
            {
                'params': self.tx,
                'lr': 3e-2
            },
            {
                'params': self.ty,
                'lr': 3e-2
            },
            {
                'params': self.s,
                'lr': 3e-2
            },
        ]
        
        optimizer = optim.Adam(variable_dict, betas=(0.5, 0.999))

        losses = [('VGG', 1, PerceptualLoss(reduce=False))]

        reg_creterion = torch.nn.MSELoss(reduce=False)
        self.opt.n_iter = iter
        loss_history = np.zeros((self.gt_images.shape[0], self.opt.n_iter, len(losses) + 1))
        state_history = np.zeros((self.gt_images.shape[0], self.opt.n_iter, 6 + self.opt.z_dim))
        image_history = []
        from torchviz import make_dot
        for iter in range(self.opt.n_iter):

            optimizer.zero_grad()

            angle = 180 * torch.cat([self.ax, self.ay, torch.zeros_like(self.az)], dim=1)
            
            fake_B = self.model.netG(self.z, angle)
            # g = make_dot(fake_B)
            # g.view()
            fake_B = self.warp_image(fake_B, self.az, self.s, self.tx,
                                    self.ty, grad=True)
            
            fake_B_upsampled = F.interpolate(fake_B,
                                             size=real_B.shape[-1],
                                             mode='bilinear')

            error_all = 0
            for l, (name, weight, criterion) in enumerate(losses):
                error = weight * \
                    criterion(fake_B_upsampled, real_B).view(
                        1, -1).mean(1)
                loss_history[:, iter, l] = error.data.cpu().numpy()
                error_all = error_all + error

            error = self.opt.lambda_reg * \
                reg_creterion(self.z, torch.zeros_like(self.z)).view(
                    1, -1).mean(1)
            loss_history[:, iter, l + 1] = error.data.cpu().numpy()
            error_all = error_all + error
            error_all.backward()
            optimizer.step()
            image_history.append(fake_B)

            state_history[:, iter, :3] = 180 * \
                torch.cat([-self.ay-0.5, self.ax+1, -self.az], dim=-1).data.cpu().numpy()
            state_history[:, iter, 3:] = torch.cat([self.tx, self.ty, self.s, self.z],
                                                   dim=-1).data.cpu().numpy()
        return state_history, loss_history, image_history