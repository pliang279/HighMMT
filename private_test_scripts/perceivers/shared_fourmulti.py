import sys
import os
import copy
import random
sys.path.insert(1,os.getcwd())
#from perceiver_pytorch.multi_modality_perceiver import MultiModalityPerceiver, InputModality
from private_test_scripts.perceivers.crossattnperceiver import MultiModalityPerceiver, InputModality

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from datasets.mimic.get_data import get_dataloader
trains1,valid1,test1=get_dataloader(7,imputed_path='/home/paul/yiwei/im.pk',no_robust=True,batch_size=20)
from datasets.avmnist.get_data import get_dataloader
trains2,valid2,test2=get_dataloader('/home/paul/yiwei/avmnist/_MFAS/avmnist',no_robust=True,unsqueeze_channel=False,to4by4=True,fracs=1)
from datasets.affect.get_data import get_simple_processed_data
trains3,valid3,test3=get_simple_processed_data('/home/paul/yiwei/MultiBench/mosei_senti_data.pkl',fracs=1,repeats=1)
from private_test_scripts.perceivers.humorloader import get_dataloader
trains4,valid4,test4=get_dataloader(1,32,1)
#trains4,valid4,test4=get_dataloader('/home/paul/MultiBench/mosi_raw.pkl',raw_path='/home/paul/MultiBench/mosi.hdf5',batch_size=3,no_robust=True)
device='cuda:0'
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
    input_channels=16,
    input_axis=2,
    num_freq_bands=6,
    max_freq=1
)
audio_spec_modality=InputModality(
    name='audiospec',
    input_channels=256,
    input_axis=2,
    num_freq_bands=6,
    max_freq=1
)

feature1_modality=InputModality(
    name='feature1',
    input_channels=35,
    input_axis=1,
    num_freq_bands=3,
    max_freq=1
)
feature2_modality=InputModality(
    name='feature2',
    input_channels=74,
    input_axis=1,
    num_freq_bands=3,
    max_freq=1
)
feature3_modality=InputModality(
    name='feature3',
    input_channels=300,
    input_axis=1,
    num_freq_bands=3,
    max_freq=1
)
feature4_modality=InputModality(
    name='feature4',
    input_channels=371,
    input_axis=1,
    num_freq_bands=3,
    max_freq=1
)
feature5_modality=InputModality(
    name='feature5',
    input_channels=81,
    input_axis=1,
    num_freq_bands=3,
    max_freq=1
)

# Specify parameter share groupings
uni_share_map = {
        'static'        : 0,
        'timeseries'    : 2,
        'colorlessimage': 1,
        'audiospec'     : 1,
        'feature1'      : 0,
        'feature2'      : 3,
        'feature3'      : 2,
        'feature4'      : 0,
        'feature5'      : 3
        }

cross_share_map = {
        ('static', 'timeseries')       : 0,
        ('colorlessimage', 'audiospec'): 1,
        ('feature1', 'feature2')       : 2,
        ('feature1', 'feature3')       : 0,
        ('feature2', 'feature3')       : 4,
        ('feature4', 'feature5')       : 3,
        ('feature4', 'feature3')       : 0,
        ('feature5', 'feature3')       : 3
    }

#join 1,3 with 43/45

num_uni_gps = 4
num_cross_gps = 5
reg = True
reg_weights = [0.003, 0.011, 0.003, 0.00]
train_weights = [1.0, 1.4, 1.1, 1.1]
weight_decay = 0.001
epochs=40
lr = 0.0001

# Specify a pretrained model path, put None if don't want to finetune from a model checkpoint 
quart = 'private_test_scripts/perceivers/fhundricmic30_small.pt'

pretrained = torch.load(quart).to(device)

for ii in range(1):

    models = []
    for i in range(max(num_uni_gps, num_cross_gps)):
        #"""
        
        m =  MultiModalityPerceiver(
            modalities=(static_modality,timeseries_modality,colorless_image_modality,audio_spec_modality,feature1_modality,feature2_modality,feature3_modality,feature4_modality,feature5_modality),
            depth=1,  # depth of net, combined with num_latent_blocks_per_layer to produce full Perceiver
            num_latents=10,
            # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim=16,  # latent dimension
            cross_heads=1,  # number of heads for cross attention. paper said 1
            latent_heads=6,  # number of heads for latent self attention, 8
            cross_dim_head=16,
            latent_dim_head=16,
            num_classes=1,  # output number of classes
            attn_dropout=0.,
            ff_dropout=0.,
            embed=True,
            weight_tie_layers=True,
            num_latent_blocks_per_layer=1, # Note that this parameter is 1 in the original Lucidrain implementation,
            cross_depth=1
        ).to(device)
        
        tmods = 32
        thmods = 96

        m.to_logits = torch.nn.Sequential(torch.nn.LayerNorm(thmods),torch.nn.Linear(thmods,2))
        m.to_logitslist=torch.nn.ModuleList([torch.nn.Sequential(torch.nn.LayerNorm(tmods),torch.nn.Linear(tmods,2)),torch.nn.Sequential(torch.nn.LayerNorm(tmods),torch.nn.Linear(tmods,10)),torch.nn.Sequential(torch.nn.LayerNorm(thmods),torch.nn.Linear(thmods,2)),torch.nn.Sequential(torch.nn.LayerNorm(thmods),torch.nn.Linear(thmods,2))]).to(device)
       
        if pretrained != None:
            m.load_state_dict(copy.deepcopy(pretrained.state_dict())) 

        models.append(m)

    model = torch.nn.ModuleList(models)

    from private_test_scripts.perceivers.train_structure_multitask_pshare import train

    records=train(model,
                 epochs,
                 [trains1, trains2, trains3, trains4],
                 [valid1, valid2, valid3, valid4],
                 [test1, test2, test3, test4],
                 [['static','timeseries'], ['colorlessimage', 'audiospec'], ['feature1', 'feature2', 'feature3'], ['feature4', 'feature5', 'feature3']],
                 'private_test_scripts/perceivers/psharetest.pt', 
                 uni_share_map, 
                 cross_share_map, 
                 num_uni_gps, 
                 num_cross_gps, 
                 one_optim=True, 
                 lr=lr,
                 device=device,
                 train_weights=train_weights,
                 is_affect=[False, False, False, False], 
                 unsqueezing=[True, False, False, False],
                 transpose=[False, False, False, False],
                 is_push=[False, False, False, False],
                 evalweights=[1,1,1,1],
                 start_from=0,
                 weight_decay=weight_decay,
                 debug=False,
                 lrs=None,
                 reg_weights=reg_weights,
                 regularise=reg)
    print(uni_share_map, cross_share_map)
    print(reg)
    print(reg_weights)
    print(epochs)
    print(lr)
    print(train_weights)
    print(weight_decay)
