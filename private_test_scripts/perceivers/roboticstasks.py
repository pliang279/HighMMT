import copy
import sys
import os
sys.path.append(os.getcwd())
# from perceiver_pytorch.multi_modality_perceiver import MultiModalityPerceiver, InputModality
from private_test_scripts.perceivers.crossattnperceiver import MultiModalityPerceiver, InputModality

import torch
from torch import nn
torch.multiprocessing.set_sharing_strategy('file_system')
from unimodals.common_models import Reshape


output_file = sys.argv[1]
print(f'Output will be {output_file}')

# Load your datasets like this
# Note that you should try to set dataset batchsize such that the training set from each dataset has approximately the same number of batches
# For example, since avmnist has about twice as much training datapoints as mimic, I used batch size 40 for avmnist and batch size 20 for mimic, so the total number of batches are approximately the same
# When training, a batch from each dataset will be run and the final loss will be a weighted sum of loss from each dataset/task

from datasets.gentle_push.data_loader import PushTask
import fannypack
fannypack.data.set_cache_path('datasets/gentle_push/cache')
trains3, valid3, test3 = PushTask.get_dataloader(16, batch_size=18, drop_last=True, test_multimodal_only=True, test_noises=[0])
# To get a subset of gentle_push data:
# trains3, valid3, test3 = PushTask.get_dataloader(16, batch_size=10, drop_last=True, test_multimodal_only=True, test_noises=[0],
#     train_sampler=torch.utils.data.SubsetRandomSampler(range(SUBSET_SIZE)))
test3 = test3['multimodal'][0]

print(len(trains3))

from datasets.robotics.get_data import get_data as robotics_get_data
from examples.robotics.robotics_utils import set_seeds as robotics_set_seeds
import yaml

with open('private_test_scripts/perceivers/robotics_training_default.yaml') as f:
    configs = yaml.load(f)
robotics_set_seeds(42, True)
trains4, valid4 = robotics_get_data('cuda', configs, '')
# To get a subset of robotics data:
# trains4, valid4 = robotics_get_data('cuda', configs, '', train_subset_factor=SUBSET_FACTOR, train_batch_size=BATCH_SIZE)
test4 = copy.deepcopy(valid4) # No test data for this dataset

print(len(trains4))

#choose your device
device='cuda:0'

# define your modalities (same way as regular perceiver)
static_modality=InputModality(
    name='static',
    input_channels=1,
    input_axis=1,
    num_freq_bands=6,
    max_freq=1
)
timeseries_modality=InputModality(
    name='timeseries',
    input_channels=1,
    input_axis=2,
    num_freq_bands=6,
    max_freq=1
)
colorless_image_modality=InputModality(
    name='colorlessimage',
    input_channels=1,
    input_axis=2,
    num_freq_bands=6,
    max_freq=1
)
audio_spec_modality=InputModality(
    name='audiospec',
    input_channels=1,
    input_axis=2,
    num_freq_bands=6,
    max_freq=1
)
timeseries_gripper_pos_modality=InputModality(
    name='timeseries_gripper_pos',
    input_channels=3,
    input_axis=1,
    num_freq_bands=6,
    max_freq=1
)
timeseries_gripper_sensors_modality=InputModality(
    name='timeseries_gripper_sensors',
    input_channels=7,
    input_axis=1,
    num_freq_bands=6,
    max_freq=1
)
timeseries_control_modality=InputModality(
    name='timeseries_control',
    input_channels=7,
    input_axis=1,
    num_freq_bands=6,
    max_freq=1
)
colorless_image_timeseries_modality=InputModality(
    name='colorlessimage_timeseries',
    input_channels=1,
    input_axis=3,
    num_freq_bands=6,
    max_freq=1
)
image_modality=InputModality(
    name='image',
    input_channels=3,
    input_axis=2,
    num_freq_bands=6,
    max_freq=1
)
force_modality=InputModality(
    name='force',
    input_channels=32,
    input_axis=1,
    num_freq_bands=6,
    max_freq=1
)
proprio_modality=InputModality(
    name='proprio',
    input_channels=8,
    input_axis=1,
    num_freq_bands=6,
    max_freq=1
)
depth_modality=InputModality(
    name='depth',
    input_channels=1,
    input_axis=2,
    num_freq_bands=6,
    max_freq=1
)
action_modality=InputModality(
    name='action',
    input_channels=4,
    input_axis=1,
    num_freq_bands=6,
    max_freq=1
)

# define your models (note that the current depth and num_latent_blocks_per_layer is reduced already)
model = MultiModalityPerceiver(
    modalities=(static_modality,timeseries_modality,colorless_image_modality,audio_spec_modality,
                timeseries_gripper_pos_modality,timeseries_gripper_sensors_modality,
                timeseries_control_modality,colorless_image_timeseries_modality,
                image_modality, force_modality, proprio_modality, depth_modality, action_modality,
                ),
    depth=1,  # depth of net, combined with num_latent_blocks_per_layer to produce full Perceiver
    num_latents=20,
    # number of latents, or induced set points, or centroids. different papers giving it different names
    latent_dim=64,  # latent dimension
    cross_heads=1,  # number of heads for cross attention. paper said 1
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

# build your classification heads, one for each dataset/task
model.to_logitslist=nn.ModuleList([
    nn.Sequential(nn.LayerNorm(64 * 3 * 4), nn.Linear(64 * 3 * 4, 16 * 2), Reshape([-1, 16, 2])).to(device),
    nn.Sequential(nn.LayerNorm(64 * 4 * 5), nn.Linear(64 * 4 * 5, 1)).to(device),
])

criteria = [
    nn.MSELoss(),
    nn.BCEWithLogitsLoss(),
]
valid_criteria = [
    nn.MSELoss(),
    lambda x, y: -torch.mean(((x >= 0) == y).float()),
]
test_criteria = valid_criteria

def encoder(x):
    if 'proprio' in x:
        x['proprio'] = x['proprio'].unsqueeze(1)
    if 'action' in x:
        x['action'] = x['action'].unsqueeze(1)
    return x

# train the model
# the dataloaders needs to be inputed as lists, and the modalities for each dataset needs to be specified as a list of lists, as shown in the example below
# the training weights are how much the losses from each input dataset are weighted, it can (and should) be tuned to produce best outcome
from private_test_scripts.perceivers.train_structure_multitask_robotics import train
train(model,100,[trains3, trains4],[valid3, valid4],[test3, test4],
    [['timeseries_gripper_pos','timeseries_gripper_sensors','colorlessimage_timeseries','timeseries_control'],
     ['image','force','proprio','depth', 'action']],
    output_file,
    encoder=encoder,
    is_affect=[False,False],transpose=[False,False],unsqueezing=[False,False],
    is_classification=[False,False],
    criterions=criteria,valid_criterions=valid_criteria,test_criterions=test_criteria,
    lr=0.0005,device=device,train_weights=[100.0, 1.0],valid_weights=[100.0, 1.0]) # Set weights here
