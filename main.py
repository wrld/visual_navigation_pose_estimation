import numpy as np
import torch
import os
import os
import torch
import numpy as np
import tqdm
import wandb
from config import load_options
from runner import eval_nocs, train_one_epoch, eval_one_epoch
from utils.loss import SetCriterion 
from models.nocs_gym import nocs_gym
from sac_2.SAC import SAC
def main(args, parser):
    # Initialization save & log
    runner_name = args.name
    save_path = os.path.join(args.save_folder, runner_name)
    os.makedirs(save_path, exist_ok=True)
    
    device = torch.device('cuda')
    if args.log == True:
        run = wandb.init(project=args.name, group=args.name)
        run.config.data = save_path
        run.name = run.id
        
    # Initialize criterions
    criterion = SetCriterion(args)
    
    # Initialize datasets
    print("========load nocs image generator==========")
    env = nocs_gym(args, parser, criterion) 
    # Initialize agent
    agent = SAC(args.image_size, env.action_space, args, save_path, device)
    if args.pretrain is not None:
        agent.policy.load_state_dict(torch.load(args.pretrain))
            
    min_test = np.infty
    args.save_path = save_path
    # Evaluation on synthetic dataset
    if args.eval==1:
        eval_one_epoch(args, agent, env, episodes=args.batch_size)
        return
    # Evaluation on real dataset
    elif args.eval ==2:
        eval_nocs(args, agent, env)
        return
        
    for i_episode in tqdm.tqdm(range(1, args.episode_nums)):
        
        train_one_epoch(args, agent, env, i_episode)
    
        if i_episode % args.save_interval == 0:
            avg_loss = eval_one_epoch(args, agent, env, i_episode)
            path = os.path.join(save_path, "checkpoint_" + str(i_episode) + ".pt")
            torch.save(agent.policy.state_dict(), path)
            path = os.path.join(save_path, "latest.pt")
            torch.save(agent.policy.state_dict(), path)

            if avg_loss < min_test:
                min_test = avg_loss
                path = os.path.join(save_path, "best_model")
                torch.save(agent.policy.state_dict(), path)
                print("save best model on ", i_episode)

if __name__ == '__main__':
    args, parser = load_options()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    main(args, parser)
