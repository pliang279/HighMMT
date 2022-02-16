import sys
import os
sys.path.insert(1,os.getcwd())
#from perceiver_pytorch.multi_modality_perceiver import MultiModalityPerceiver, InputModality
from private_test_scripts.perceivers.crossattnperceiver import MultiModalityPerceiver, InputModality

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from datasets.mimic.get_data import get_dataloader
trains1,valid1,test1=get_dataloader(7,imputed_path='/home/pliang/yiwei/im.pk',no_robust=True,batch_size=20,fracs=1)
from datasets.avmnist.get_data import get_dataloader
trains2,valid2,test2=get_dataloader('/home/pliang/yiwei/avmnist/_MFAS/avmnist',no_robust=True,unsqueeze_channel=False,to4by4=True,fracs=1)
from datasets.affect.get_data import get_simple_processed_data
trains3,valid3,test3=get_simple_processed_data('/home/pliang/yiwei/MultiBench/mosei_senti_data.pkl',fracs=1,repeats=1)
#from private_test_scripts.perceivers.humorloader import get_dataloader
#trains4,valid4,test4=get_dataloader(1,32,1)
#trains4,valid4,test4=get_dataloader('/home/paul/MultiBench/mosi_raw.pkl',raw_path='/home/paul/MultiBench/mosi.hdf5',batch_size=3,no_robust=True)
device='cuda:3'
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
for i in range(1):
    #"""
    model = MultiModalityPerceiver(
        modalities=(static_modality,timeseries_modality,colorless_image_modality,audio_spec_modality,feature1_modality,feature2_modality,feature3_modality,feature4_modality,feature5_modality),
        depth=1,  # depth of net, combined with num_latent_blocks_per_layer to produce full Perceiver
        num_latents=20,
        # number of latents, or induced set points, or centroids. different papers giving it different names
        latent_dim=64,  # latent dimension
        cross_heads=1,  # number of heads for cross attention. paper said 1
        latent_heads=6,  # number of heads for latent self attention, 8
        cross_dim_head=64,
        latent_dim_head=64,
        num_classes=1,  # output number of classes
        attn_dropout=0.,
        ff_dropout=0.,
        #embed=True,
        weight_tie_layers=True,
        num_latent_blocks_per_layer=1, # Note that this parameter is 1 in the original Lucidrain implementation,
        cross_depth=1
    ).to(device)
    model.to_logitslist=torch.nn.ModuleList([torch.nn.Sequential(torch.nn.LayerNorm(128),torch.nn.Linear(128,2)),torch.nn.Sequential(torch.nn.LayerNorm(128),torch.nn.Linear(128,10)),torch.nn.Sequential(torch.nn.LayerNorm(384),torch.nn.Linear(384,2))]).to(device)

    from private_test_scripts.perceivers.train_structure_multisave import train

    records=train(model,25,[trains1,trains2,trains3],[valid1,valid2,valid3],[test1,test2,test3],[['static','timeseries'],['colorlessimage','audiospec'],['feature1','feature2','feature3']],'private_test_scripts/perceivers/threesave/a.pt',lr=0.0008,device=device,train_weights=[1.2,0.9,1.1],is_affect=[False,False,False],unsqueezing=[True,False,False],transpose=[False,False,False],evalweights=[1,1,1],start_from=0,weight_decay=0.001)
