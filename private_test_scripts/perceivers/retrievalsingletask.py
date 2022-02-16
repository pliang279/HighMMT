import sys
import os
sys.path.insert(1,os.getcwd())
#from perceiver_pytorch.multi_modality_perceiver import MultiModalityPerceiver, InputModality
from private_test_scripts.perceivers.crossattnperceiver import MultiModalityPerceiver, InputModality

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from private_test_scripts.perceivers.get_retrieval_data import get_complex_data
trains1,valid1,test1=get_complex_data()
#from datasets.avmnist.get_data import get_dataloader
#trains2,valid2,test2=get_dataloader('/home/pliang/yiwei/avmnist/_MFAS/avmnist',no_robust=True,unsqueeze_channel=False,to4by4=True,fracs=1)
device='cuda:1'
colorful_image_modality=InputModality(
    name='colorfulimage',
    input_channels=48,
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
for i in range(1):
    #"""
    model = MultiModalityPerceiver(
        modalities=(colorful_image_modality,colorless_image_modality,audio_spec_modality),
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
    #"""
    #model=torch.load('private_test_scripts/perceivers/fhundricmic6.pt').to(device)
    #for p in model.layers:
     #   p.requires_grad=False
    model.to_logitslist=torch.nn.ModuleList([torch.nn.Sequential(torch.nn.LayerNorm(128),torch.nn.Linear(128,2))]).to(device)
    
    from private_test_scripts.perceivers.train_structure_multitask import train

    records=train(model,250,[trains1],[valid1],[test1],[['colorfulimage','audiospec']],'private_test_scripts/perceivers/fhundricmi1.pt',lr=0.0008,device=device,train_weights=[1.0],is_affect=[False],unsqueezing=[False],transpose=[False],evalweights=[1],start_from=0,weight_decay=0.0,optimizer=torch.optim.Adam)
