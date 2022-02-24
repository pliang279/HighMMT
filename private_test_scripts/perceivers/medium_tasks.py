import sys
import os
sys.path.append(os.getcwd())
# from perceiver.perceiver_pytorch.multi_modality_perceiver import MultiModalityPerceiver, InputModality
from private_test_scripts.perceivers.crossattnperceiver import MultiModalityPerceiver, InputModality

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

isdebug=False

# enrico
from datasets.enrico.get_data import get_dataloader
dls, weights = get_dataloader('/home/pliang/shentong/data/enrico/dataset', batch_size=32, isdebug=isdebug)
trains1, valid1, test1 = dls
test1 = test1['image'][0]
print("loaded enrico dataset successfully!!!")

# avmnist
from datasets.avmnist.get_data import get_dataloader
trains2,valid2,test2=get_dataloader('/home/pliang/shentong/data/avmnist',flatten_audio=True, unsqueeze_channel=1, batch_size=32, isdebug=isdebug)
print("loaded avmnist dataset successfully!!!")

from datasets.gentle_push.data_loader import PushTask
import argparse
import fannypack

Task = PushTask
# Parse args
parser = argparse.ArgumentParser()
Task.add_dataset_arguments(parser)
args = parser.parse_args()
dataset_args = Task.get_dataset_args(args)

fannypack.data.set_cache_path('/home/pliang/shentong/data/gentle_push/cache')

trains3,valid3,test3 = Task.get_dataloader(16, batch_size=32, drop_last=True)
test3 = test3['image'][0]
print("loaded gentle_push dataset successfully!!!")



device='cuda'

# gentle_push modality
# [B, 16, 3, 1]
pose_modality = InputModality(
    name='pose',
    input_channels=1,  # number of channels for each token of the input
    input_axis=1,  # number of axes, 2 for images
    num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
    max_freq=4.,  # maximum frequency, hyperparameter depending on how fine the data is
)
# [B, 16, 7, 1]
sensor_modality = InputModality(
    name='sensor',
    input_channels=1,  # number of channels for mono audio
    input_axis=1,  # number of axes, 2 for images
    num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
    max_freq=8.,  # maximum frequency, hyperparameter depending on how fine the data is
)

# [B, 16, 32, 32, 1]
trajectory_modality = InputModality(
    name='trajectory',
    input_channels=1,  # number of channels for each token of the input
    input_axis=2,  # number of axes, 2 for images
    num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
    max_freq=4.,  # maximum frequency, hyperparameter depending on how fine the data is
)

# [B, 16, 7, 1]
control_modality = InputModality(
    name='control',
    input_channels=1,  # number of channels for each token of the input
    input_axis=1,  # number of axes, 2 for images
    num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
    max_freq=4.,  # maximum frequency, hyperparameter depending on how fine the data is
)
# enrico modality
image_1_modality = InputModality(
    name='image_1',
    input_channels=3,  # number of channels for each token of the input
    input_axis=2,  # number of axes, 2 for images
    num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
    max_freq=4.,  # maximum frequency, hyperparameter depending on how fine the data is
)
image_2_modality = InputModality(
    name='image_2',
    input_channels=3,  # number of channels for mono audio
    input_axis=2,  # number of axes, 2 for images
    num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
    max_freq=8.,  # maximum frequency, hyperparameter depending on how fine the data is
)

image_modality = InputModality(
    name='image',
    input_channels=1,  # number of channels for each token of the input
    input_axis=2,  # number of axes, 2 for images
    num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
    max_freq=4.,  # maximum frequency, hyperparameter depending on how fine the data is
)
audio_modality = InputModality(
    name='audio',
    input_channels=1,  # number of channels for mono audio
    input_axis=1,  # number of axes, 2 for images
    num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
    max_freq=8.,  # maximum frequency, hyperparameter depending on how fine the data is
)
model = MultiModalityPerceiver(
    modalities=(image_1_modality,image_2_modality,image_modality,audio_modality,pose_modality,sensor_modality,trajectory_modality,control_modality),
    depth=1,  # depth of net, combined with num_latent_blocks_per_layer to produce full Perceiver
    cross_heads=1,  # number of heads for cross attention. paper said 1
    num_latents=12,
    # number of latents, or induced set points, or centroids. different papers giving it different names
    latent_dim=64,  # latent dimension
    latent_heads=8,  # number of heads for latent self attention, 8
    cross_dim_head=64,
    latent_dim_head=64,
    num_classes=2,  # output number of classes
    attn_dropout=0.,
    ff_dropout=0.,
    weight_tie_layers=True,
    num_latent_blocks_per_layer=1  # Note that this parameter is 1 in the original Lucidrain implementation
    # whether to weight tie layers (optional, as indicated in the diagram)
).to(device)

model.to_logitslist=torch.nn.ModuleList([torch.nn.Sequential(torch.nn.LayerNorm(128),torch.nn.Linear(128,20)).to(device),torch.nn.Sequential(torch.nn.LayerNorm(128),torch.nn.Linear(128,10)).to(device),torch.nn.Sequential(torch.nn.LayerNorm(768),torch.nn.Linear(768,2)).to(device)])

from private_test_scripts.perceivers.train_structure_multitask import train, single_test

train(model,100,[trains1,trains2,trains3],[valid1,valid2,valid3],[test1,test2,test3],\
    [['image_1','image_2'],['image','audio'],['pose', 'sensor', 'trajectory', 'control']],\
    './model.pt',lr=0.001,device=device,train_weights=[1.0,1.2,1.0],\
    is_affect=[False,False,False],is_push=[False, False, True], unsqueezing=[False,False,False],transpose=[False,False,False])
