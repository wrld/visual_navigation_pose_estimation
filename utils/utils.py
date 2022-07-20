import math
import torch
import torchvision.models.vgg as models
from torchvision import transforms
import numpy as np
from scipy.linalg import logm, norm
import matplotlib.pyplot as plt
import os
degree_thres_list = range(0, 61, 1)
def is_between(a, x, b):
        return min(a, b) <= x <= max(a, b)
    
def rotationMatrixToEulerAngles(R) :
        x = -math.asin(-R[2, 1])
        y = -math.atan2(R[2, 0], R[2, 2])
        z = -math.atan2(R[0, 1], R[1, 1])
        return np.array([y, x, z])
def calc_ap(loss):
    
    acc = np.zeros_like(degree_thres_list)
    for j in range(len(loss)):
            for k in range(len(degree_thres_list)):
                    if int(loss[j]) <= degree_thres_list[k]:
                            acc[k] += 1.0
    acc = acc / len(loss) * 100.0
    return acc

def show_AP(angle_loss, name, path):
    
    acc_il_step_1 = calc_ap(angle_loss)
    plt.figure(figsize=(5,5))
    plt.plot(degree_thres_list,
            acc_il_step_1,
            linestyle="-",
            color=(0, 0, 1, 0.2),
            # marker="o",
            alpha=0.5,
            linewidth=3,
            label=name)
    plt.savefig(os.path.join(path, "AP_0_60_degree.png"))
    
def load_image(path, size = 128):
    transform_list = []
    target_size = [size, size]
    transform_list.append(transforms.Resize(target_size, Image.BICUBIC))

    transform_list += [transforms.ToTensor()]

    image = Image.open(path).convert('RGB')
    trans = transforms.Compose(transform_list)
    image = trans(image)
    return image
def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


