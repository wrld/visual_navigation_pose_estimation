import numpy as np
import torch
import os
from torchvision.utils import save_image, make_grid
import os
import torch
import numpy as np
import tqdm
import pickle
from PIL import Image  
import cv2
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as scipy_rot
import wandb
from PIL import Image
from torchvision import transforms
def train_one_epoch(args, agent, env, i_episode):
    
    state = env.reset(batch_size=args.batch_size)
    for i in range(args.max_step):
        agent.policy_optim.zero_grad()
        # sample action from gaussian policy
        action = agent.select_action(state)
        state = env.step(action, eval=False)
        loss = env.losses
        loss.backward()
        agent.policy_optim.step()
    
    if args.log == True and i_episode % args.log_interval == 0:
        # evaluation for the rotation error and translation error
        eval_loss = env.evaluate()
        eval_log = {
                    "train/" + k: (eval_loss[k]).mean()
                    for k in eval_loss.keys()
                }
        state = state.reshape(-1, state.shape[2], state.shape[3], state.shape[4])
        grid_image = make_grid(state, nrow=int(np.sqrt(state.shape[0])))
        ndarr = grid_image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        img = Image.fromarray(ndarr)
        # Log to wandb
        img = wandb.Image(img)
        wandb.log(env.log_losses, step=i_episode)
        wandb.log(eval_log, step=i_episode)
        wandb.log({"train/state": img}, step=i_episode)
        
# Evaluation on synthetic dataset
def eval_one_epoch(args, agent, env, i_episode=0, episodes=10):
    total_loss = 0
    state_history = []
    
    state = env.reset(batch_size = episodes)
    for i in range(args.max_step):
        state_history.append(state[:, 1, :, :, :].data)
        action = agent.select_action(state, True)
        state = env.step(action, eval=True).clone()
    
    total_loss += env.losses 
    state_history.append(state[:, 1, :, :, :].data)
    # Start GD optimization after IL
    if args.gd_optimize == True:
        _, _, image_history = env.optimize(iter=10)
        state_history.append(image_history[len(image_history)-1])
    state_history.append(env.gt_images)
    
    eval_loss = env.evaluate()
    avg_loss = total_loss / episodes 
    state_history = torch.stack(state_history).reshape(-1, 3, 64, 64)
    grid_image = make_grid(state_history, nrow=episodes)
    ndarr = grid_image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    img = Image.fromarray(ndarr)
    print("===========Test Results=============")
    print("Rotation error: {} degree, \nTranslation error: {} cm".format(eval_loss['rot_distance'].mean(), eval_loss['trans_distance'].mean()))
    img.save(os.path.join(args.save_path, "eval_" + str(episodes) + ".png"))
    if args.log == True:
        eval_log = {
                "test/" + k: (eval_loss[k]).mean()
                for k in eval_loss.keys()
            }
        img = wandb.Image(img)
        wandb.log(env.log_losses, step=i_episode)
        wandb.log({"test/avg loss": avg_loss}, step=i_episode)
        wandb.log(eval_log, step=i_episode)
        wandb.log({"test/state": img}, step=i_episode)
    return avg_loss

# Evaluation on real dataset
def eval_nocs(args, agent, env):
    intrinsics = np.array(
        [[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
    # Rendering parameters
    focal_lengh_render = 70.
    image_size_render = 64

    # Average scales from the synthetic training set CAMERA
    mean_scales = np.array([0.34, 0.21, 0.19, 0.15, 0.46, 0.17])
    
    env.opt.dataroot = './datasets/test/'
    nocs_list = sorted(os.listdir(os.path.join(
        './datasets/test/', 'nocs_det')))[::env.opt.skip]

    interval = len(nocs_list)//(env.opt.num_agent -
                                1) if env.opt.num_agent > 1 else len(nocs_list)
    task_range = nocs_list[interval*env.opt.id_agent:min(
        interval*(env.opt.id_agent+1), len(nocs_list))]
    
    output_folder = args.save_path
    print("Starting evaluation on nocs for ", env.category)
    image_all  = []
    eval_index = 0
    for file_name in tqdm.tqdm(task_range):
        file_path = os.path.join(env.opt.dataroot, 'nocs_det', file_name)
        pose_file = pickle.load(open(file_path, 'rb'), encoding='utf-8')

        image_name = pose_file['image_path'].replace(
            'data/real/test', env.opt.dataroot+'/real_test/')+'_color.png'
        image = cv2.imread(image_name)[:, :, ::-1]

        masks = pose_file['pred_mask']
        bboxes = pose_file['pred_bboxes']

        pose_file['pred_RTs_ours'] = np.zeros_like(pose_file['pred_RTs'])
        class_pred = env.categories.index(env.opt.category)+1
        if class_pred in pose_file['pred_class_ids']:
            label = [i for i, e in enumerate(
                pose_file['pred_class_ids']) if e == class_pred]

            for id in label:
                eval_index += 1
                bbox = bboxes[id]
                image_mask = image.copy()
                image_mask[masks[:, :, id] == 0, :] = 255
                image_mask = image_mask[bbox[0]:bbox[2], bbox[1]:bbox[3], :]
                # A = transforms.ToTensor()(Image.fromarray(cv2.cvtColor(image_mask,cv2.COLOR_BGR2RGB))).unsqueeze(0)
                
                A = (torch.from_numpy(image_mask.astype(np.float32)).cuda(
                ).unsqueeze(0).permute(0, 3, 1, 2) / 255) * 2 - 1
                _, c, h, w = A.shape
                s = max(h, w) + 30
                A = F.pad(A, [(s - w)//2, (s - w) - (s - w)//2,
                                (s - h)//2, (s - h) - (s - h)//2], value=1)
                A = F.interpolate(
                    A, size=env.opt.target_size, mode='bilinear')
                A = A.to(env.model.device)
                
                state = env.reset(real=A.clone())
                
                for i in range(args.max_step):
                    image_all.append(state[:, 1, :, :, :])
                    action = agent.select_action(state, True)
                    state = env.step(action, eval=True)
                image_all.append(state[:, 1, :, :, :])
                if args.gd_optimize == True:
                    state_history, loss_history, image_history = env.optimize(iter=10)
                    image_all.append(image_history[len(image_history)-1])
                    states = state_history[-1, -1, :]
                else:
                    states_angle = 180 * \
                        torch.cat([-env.ay-0.5, env.ax+1, -env.az], dim=-1)
                    states_other = torch.cat([env.tx, env.ty, env.s, env.z], dim=-1)
                    states = torch.cat([states_angle, states_other], dim=-1).data.squeeze(0).cpu().numpy()
                    

                image_all.append(env.gt_images)

                pose_file['pred_RTs_ours'][id][:3, :3] = scipy_rot.from_euler(
                        'yxz', states[:3], degrees=True).as_dcm()[:3, :3]
                angle = -states[2] / 180 * np.pi
                mat = np.array([[states[5]*np.cos(angle), -states[5]*np.sin(angle), states[5]*states[3]],
                                [states[5]*np.sin(angle),  states[5]*np.cos(
                                    angle), states[5]*states[4]],
                                [0,                         0,                 1]])

                mat_inv = np.linalg.inv(mat)
                u = (bbox[1] + bbox[3])/2 + mat_inv[0, 2]*s/2
                v = (bbox[0] + bbox[2])/2 + mat_inv[1, 2]*s/2

                z = image_size_render/(s/states[5]) * (
                    intrinsics[0, 0]+intrinsics[1, 1])/2 / focal_lengh_render * mean_scales[class_pred-1]

                pose_file['pred_RTs_ours'][id][2, 3] = z
                pose_file['pred_RTs_ours'][id][0, 3] = (
                    u - intrinsics[0, 2])/intrinsics[0, 0]*z
                pose_file['pred_RTs_ours'][id][1, 3] = (
                    v - intrinsics[1, 2])/intrinsics[1, 1]*z
                pose_file['pred_RTs_ours'][id][3, 3] = 1

            

                
            f = open(os.path.join(output_folder, file_name), 'wb')
            pickle.dump(pose_file, f, -1)
    
    image_all = torch.stack(image_all).reshape(-1, 3, 64, 64)
    grid_image = make_grid(image_all, nrow=eval_index)
    ndarr = grid_image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    img = Image.fromarray(ndarr)
    img.save(os.path.join(args.save_path, "eval_" + env.category + ".png"))
   