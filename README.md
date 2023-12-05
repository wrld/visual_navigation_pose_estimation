# [ECCV'22] A Visual Navigation Perspective for Category-Level Object Pose Estimation

This is the official repository for the ECCV 2022 paper ["A Visual Navigation Perspective for Category-Level Object Pose Estimation"](https://arxiv.org/abs/2203.13572). 
![](https://github.com/wrld/visual_navigation_pose_estimation/blob/main/images/teaser_video.gif)
## System Environments:

You can use anaconda and create an anaconda environment:

``` shell
conda env create -f environment.yml
conda activate visual_nav
```
## Datasets
We set [neural_object_fitting](https://github.com/xuchen-ethz/neural_object_fitting) as our image generator for NOCS, their checkpoints and datasets could be downloaded following:

``` shell
sh prepare_datasets.sh
```

## Download Pre-trained model

Download the pretrained models [here](https://drive.google.com/drive/folders/1WFB1fJNyJgWUdyxqKHrpsUXuUpmImhcm?usp=sharing) and put them into `./pretrained_model/`.

## Example Usage

### Training on Synthetic Dataset

Run the following command to train on specific category (can / bottle / bowl / mug / laptop / camera)
```
python main.py --dataset [category] --name [running name]
```

The saved models and evaluation results could be check at './results/'.
### Training visualize

To visualize the training process, you can run:

``` shell
# use wandb to visualize the training loss and states
python main.py --dataset [category] --log True --log_interval 50
```
Then open the wandb link to monitor the training process.
### Training options

There are several settings you can change by adding arguments below:

| Arguments           | What it will trigger                            | Default              |
| ------------------- | ----------------------------------------------- | -------------------- |
| --batch_size        | The batch size of input                         |   50                  |
| --lr                | The learning rate for training                  | 0.00003                  |
| --pretrain          | Continue to train with pretrained model         |  None                    |
| --save_interval     | save model interval                             |  1000                    |
| --episode_nums      | maximum episodes number                         |  50000                    |

### Evaluation

To evaluate on synthetic dataset based on the pretrained model, run the following command:
``` shell
python main.py --dataset [category] --eval 1 --pretrain [path] --gd_optimize True
```
The evaluation results will be reported with plot.

To evaluate on real dataset, run the following command:
``` shell
python main.py --dataset [category] --eval 2 --pretrain [path] --gd_optimize True
```

To calculate the score:
``` shell
python nocs/eval.py --dataset [category]
```
The evaluation results of specific category will be reported.

## Acknowledgement
Our code is based on [neural_object_fitting](https://github.com/xuchen-ethz/neural_object_fitting) and [pytorch-soft-actor-critic](https://github.com/pranz24/pytorch-soft-actor-critic).
