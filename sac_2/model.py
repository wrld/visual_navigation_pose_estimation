# Adaptive from https://github.com/pranz24/pytorch-soft-actor-critic
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class QNetwork(nn.Module):
    def __init__(self, num_inputs, state_len, num_actions, hidden_dim):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=3, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(num_inputs)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(num_inputs)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, state_len//2)
        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + 3, hidden_dim)
        self.linear1_ = nn.Linear(num_inputs + 19, hidden_dim)
        self.linear2_ = nn.Linear(2*hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + 3, hidden_dim)
        self.linear4_ = nn.Linear(num_inputs + 19, hidden_dim)
        self.linear5_ = nn.Linear(2*hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, input, action):
        #gt images 
        in_1 = input[:, 0, :, :, :]
        # observed images
        in_2 = input[:, 1, :, :, :]
        
        x_1 = F.relu(self.bn1(self.conv1(in_1)))
        x_1 = F.relu(self.bn2(self.conv2(x_1)))
        x_1 = F.relu(self.bn3(self.conv3(x_1)))
        x_1 = torch.flatten(x_1,1)
        x_1 = self.head(x_1)
        x_2 = F.relu(self.bn1(self.conv1(in_2)))
        x_2 = F.relu(self.bn2(self.conv2(x_2)))
        x_2 = F.relu(self.bn3(self.conv3(x_2)))
        x_2 = torch.flatten(x_2,1)
        x_2 = self.head(x_2)
        state = torch.cat([x_1, x_2], dim=1)
    
        rot_embed = torch.cat([state, action[:, 0:3]], 1)
        trans_embed = torch.cat([state, action[:, 3:22]], 1)
   
        x1 = F.relu(self.linear1(rot_embed))
        x1_ = F.relu(self.linear1_(trans_embed))
        x1 = torch.cat([x1, x1_], dim=1)
        x1 = F.relu(self.linear2_(x1))

        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(rot_embed))
        x2_ = F.relu(self.linear4_(trans_embed))
        x2 = torch.cat([x2, x2_], dim=1)
        x2 = F.relu(self.linear5_(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, state_len, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv1_1 = nn.Conv2d(3, 16, kernel_size=3, stride=2)
        self.bn1_1 = nn.BatchNorm2d(16)
        self.conv2_1 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.bn2_1 = nn.BatchNorm2d(32)
        self.conv3_1 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.bn3_1 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=3, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(num_inputs)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(num_inputs)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, state_len//2)
        self.head_1 = nn.Linear(linear_input_size, state_len//2)
        self.linear1 = nn.Linear(state_len, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, 2)
        self.log_std_linear = nn.Linear(hidden_dim, 2)
        self.linear3 = nn.Linear(state_len, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear_1 = nn.Linear(hidden_dim, 1)
        self.log_std_linear_1 = nn.Linear(hidden_dim, 1)
        self.mean_linear_ = nn.Linear(hidden_dim, 3)
        self.log_std_linear_ = nn.Linear(hidden_dim, 3)
        self.linear5 = nn.Linear(state_len, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, hidden_dim)
        self.linear7 = nn.Linear(state_len, hidden_dim)
        self.linear8 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear_2 = nn.Linear(hidden_dim, num_actions - 6)
        self.log_std_linear_2 = nn.Linear(hidden_dim, num_actions - 6)
        self.apply(weights_init_)
        self.angle_net = [
            self.conv1, self.conv2, self.conv3, self.bn1, self.bn2, self.bn3, self.head, self.linear1, self.linear2, 
            self.linear3, self.linear4, self.mean_linear,self.mean_linear_1, self.log_std_linear, self.log_std_linear_1]
        self.trans_net = [
            self.conv1_1, self.conv2_1, self.conv3_1, self.bn1_1, self.bn2_1, self.bn3_1, self.head_1, self.linear5, self.linear6, 
            self.linear7, self.linear8, self.mean_linear_,self.mean_linear_2, self.log_std_linear_, self.log_std_linear_2]

        self.action_scale = torch.tensor(1.)
        self.action_bias = torch.tensor(0.)

    def forward(self, input):
        # gt images 
        in_1 = input[:, 0, :, :, :]
        # observed images
        in_2 = input[:, 1, :, :, :]
        
        x_1 = F.relu(self.bn1(self.conv1(in_1)))
        x_1 = F.relu(self.bn2(self.conv2(x_1)))
        x_1 = F.relu(self.bn3(self.conv3(x_1)))
        x_1 = torch.flatten(x_1,1)
        x_1 = self.head(x_1)
        x_2 = F.relu(self.bn1(self.conv1(in_2)))
        x_2 = F.relu(self.bn2(self.conv2(x_2)))
        x_2 = F.relu(self.bn3(self.conv3(x_2)))
        x_2 = torch.flatten(x_2,1)
        x_2 = self.head(x_2)
        state = torch.cat([x_1, x_2], dim=1)

        x_1 = F.relu(self.bn1_1(self.conv1_1(in_1)))
        x_1 = F.relu(self.bn2_1(self.conv2_1(x_1)))
        x_1 = F.relu(self.bn3_1(self.conv3_1(x_1)))
        x_1 = torch.flatten(x_1,1)
        x_1 = self.head_1(x_1)
        x_2 = F.relu(self.bn1_1(self.conv1_1(in_2)))
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_2)))
        x_2 = F.relu(self.bn3_1(self.conv3_1(x_2)))
        x_2 = torch.flatten(x_2,1)
        x_2 = self.head_1(x_2)
        state_1 = torch.cat([x_1, x_2], dim=1)

            
        # angle
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        # az
        x = F.relu(self.linear3(state))
        x = F.relu(self.linear4(x))
        mean_1 = self.mean_linear_1(x)
        log_std_1 = self.log_std_linear_1(x)
        log_std_1 = torch.clamp(log_std_1, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        #sxy
        x = F.relu(self.linear5(state_1))
        x = F.relu(self.linear6(x))
        mean_ = self.mean_linear_(x)
        log_std_ = self.log_std_linear_(x)
        log_std_ = torch.clamp(log_std_, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        #z
        x = F.relu(self.linear7(state_1))
        x = F.relu(self.linear8(x))
        mean_2 = self.mean_linear_2(x)
        log_std_2 = self.log_std_linear_2(x)
        log_std_2 = torch.clamp(log_std_2, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        mean = torch.cat([mean,mean_1, mean_, mean_2], dim=1)
        log_std = torch.cat([log_std, log_std_1, log_std_, log_std_2], dim=1)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)

class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)


class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, input):
        x_1 = F.relu(self.bn1(self.conv1(x_1)))
        x_1 = F.relu(self.bn2(self.conv2(x_1)))
        x_1 = F.relu(self.bn3(self.conv3(x_1)))
        x_1 = self.head(x_1.view(x_1.size(0), -1))
        x_2 = F.relu(self.bn1(self.conv1(x_2)))
        x_2 = F.relu(self.bn2(self.conv2(x_2)))
        x_2 = F.relu(self.bn3(self.conv3(x_2)))
        x_2 = self.head(x_2.view(x_2.size(0), -1))
        state = torch.cat([x_1, x_2], dim=1)
        return state
