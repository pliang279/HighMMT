import sys
import os
sys.path.insert(1,os.getcwd())
#from perceiver_pytorch.multi_modality_perceiver import MultiModalityPerceiver, InputModality
from private_test_scripts.perceivers.crossattnperceiver import MultiModalityPerceiver, InputModality

from private_test_scripts.perceivers.metrics import *

import torch
from torch import nn
import numpy as np
torch.multiprocessing.set_sharing_strategy('file_system')
from datasets.mimic.get_data import get_dataloader
trains1,valid1,test1=get_dataloader(7,imputed_path='/home/paul/yiwei/im.pk',no_robust=True,batch_size=16)
from datasets.avmnist.get_data import get_dataloader
trains2,valid2,test2=get_dataloader('/home/paul/yiwei/avmnist/_MFAS/avmnist',no_robust=True,unsqueeze_channel=False,to4by4=True,fracs=1)
from datasets.affect.get_data import get_simple_processed_data
trains3,valid3,test3=get_simple_processed_data('/home/paul/yiwei/MultiBench/mosei_senti_data.pkl',fracs=1,repeats=1)
from private_test_scripts.perceivers.humorloader import get_dataloader
trains4,valid4,test4=get_dataloader(1,32,1)
#trains4,valid4,test4=get_dataloader('/home/paul/MultiBench/mosi_raw.pkl',raw_path='/home/paul/MultiBench/mosi.hdf5',batch_size=3,no_robust=True)
'''
from datasets.enrico.get_data import get_dataloader
dls, weights = get_dataloader('/home/paul/jtsaw/HighMMT/datasets/enrico/dataset', batch_size=16)
trains5, valid5, test5 = dls
test5 = test5['image'][0]
#print("loaded enrico dataset successfully!!!")

from datasets.gentle_push.data_loader import PushTask
import argparse
import fannypack
Task = PushTask
# Parse args
parser = argparse.ArgumentParser()
Task.add_dataset_arguments(parser)
args = parser.parse_args()
dataset_args = Task.get_dataset_args(args)

fannypack.data.set_cache_path('/home/paul/jtsaw/HighMMT/datasets/gentle_push/cache')

trains6,valid6,test6 = Task.get_dataloader(16, batch_size=20, drop_last=True)
test6 = test6['image'][0]
#print("loaded gentle_push dataset successfully!!!")
'''
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


unimodal_dict = {
        "mimic": [trains1, valid1, test1, ['static', 'timeseries'], 'private_test_scripts/perceivers/mimic_uni',1280, 2],
        'avmnist': [trains2, valid2, test2, ['colorlessimage', 'audiospec'], 'private_test_scripts/perceivers/avmnist_uni', 1280, 10],
        'affect': [trains3, valid3, test3, ['feature1', 'feature2', 'feature3'], 'private_test_scripts/perceivers/affect_uni', 1280, 2],
        'humor': [trains4, valid4, test4, ['feature4', 'feature5', 'feature3'], 'private_test_scripts/perceivers/humor.uni', 1280, 2]
        #'enrico': [trains5, valid5, test5, ['image_1', 'image_2'], 'private_test_scripts/perceivers/enrico_uni', 1280, 20],
        #'gentle_push': [trains6, valid6, test6, ['pose', 'sensor', 'trajectory', 'control'], 'private_test_scripts/perceivers/gpush_uni', 1280, 2]
        }


def train_unimodal_model(unimodal_dict, datasets, mods, reuse_tologits=False, savepath=None):
    #mods: 2D array where each row is modalities for each dataset 
    num = len(datasets)

    all_trains = []
    all_valids = []
    all_tests = []
    all_mods = []
    unsqueeze = []
    modules = []
    is_push = []
    save_path_net = ''
    for i, dataset in enumerate(datasets):
        trains, valid, test, mod_type, save_path, embed_size, num_class = unimodal_dict[dataset]
        all_trains.append(trains)
        all_valids.append(valid)
        all_tests.append(test)
        all_mods.append(mod_type)
        is_push.append(dataset == 'gentle_push')
        unsqueeze.append(dataset == 'mimic')
        if mods != None and len(mods[i]) == 2:
            embed_size = 128
        elif mods != None and len(mods[i]) == 3:
            embed_size = 384
        modules.append(torch.nn.Sequential(torch.nn.LayerNorm(embed_size),torch.nn.Linear(embed_size,num_class)))
        save_path_net = save_path_net + '_' + dataset + '_' + str(mods[i])

    for i in range(1):
    #"""
        model = MultiModalityPerceiver(
            modalities=(static_modality,timeseries_modality,colorless_image_modality,audio_spec_modality,feature1_modality,feature2_modality,feature3_modality,feature4_modality,feature5_modality, image_1_modality, image_2_modality, pose_modality, sensor_modality, trajectory_modality, control_modality),
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
            embed=True,
            weight_tie_layers=True,
            num_latent_blocks_per_layer=1, # Note that this parameter is 1 in the original Lucidrain implementation,
            cross_depth=1
        ).to(device)
        model.to_logitslist=torch.nn.ModuleList(modules).to(device)

        from private_test_scripts.perceivers.train_structure_multitask_tasksim import train

        if savepath != None:
            save_paths = savepath
        else: 
            save_paths = save_path_net + '.pt'
        #import copy
        #mods_train = []
        #for mod in mods:
            #mods_train.append(set(mod))
        #if mods == [['static', 'timeseries']] or mods == [['colorlessimage', 'audio']]:
            #mods_train = None
        #else:
            #mods_train = mods
        mods_train = set()
        for mod in mods[0]:
            mods_train.add(mod)
        #print(mods_train)
        records=train(model,80,all_trains,all_valids,all_tests,all_mods, save_paths, lr=0.0008,device=device,train_weights=([1.0]*num),
                is_affect=([False]*num),unsqueezing=unsqueeze,is_push=is_push, transpose=([False]*num),evalweights=([1]*num),start_from=0,weight_decay=0.001, 
                single_modality=mods_train, reuse_tologits=reuse_tologits)


######################################################## Similarity Metrics ##############################################################

def target_data_source_classifier(source_dataset, source_modality, target_dataset, target_modality, 
                            unimodal_dict, transfer_type, device="cuda:0", switch_embed=False, prov_path=None):
    _, _, test_s, _,source_path, embed_size, num_class = unimodal_dict[source_dataset]

    trains_t, valids_t, test_t, modalities_t, target_path, embed_size_t, num_class_t = unimodal_dict[target_dataset]

    model_path = source_path + '_' + source_modality + '_embed_flatten.pt'
    tmodel_path = target_path + '_' + target_modality + '_embed_flatten.pt'

    if prov_path != None: 
        model_path = prov_path

    model = torch.load(model_path).to(device)
    classifier_target = torch.load(tmodel_path).to(device)
    if switch_embed:
        model.embed = classifier_target.embed
    #fourmulti = torch.load('private_test_scripts/perceivers/fhundricmic301_embed.pt').to(device)

    unsqueezing = target_dataset == 'mimic'
    transfer_save_path = 'private_test_scripts/perceivers/tmp1.pt'

    if transfer_type=='retrain_head':
        epochs = 40
        lr = 0.0003
        weight_decay = 0.001
        model.to_logitslist=torch.nn.ModuleList([torch.nn.Sequential(torch.nn.LayerNorm(embed_size),torch.nn.Linear(embed_size,num_class_t))]).to(device)
    elif transfer_type=='finetune':
        epochs = 40
        lr = 0.0003
        weight_decay = 0.001

    from private_test_scripts.perceivers.train_structure_multitask_tasksim import train
    test_accs = train(model, epochs, [trains_t], [valids_t], [test_t], [modalities_t], transfer_save_path, lr=lr, device=device, train_weights=[1.0],
                is_affect=[False], unsqueezing=[unsqueezing], transpose=[False], evalweights=[1], start_from=0, weight_decay=weight_decay, 
                single_modality=[[target_modality]], transfer_type=transfer_type, embed_size=embed_size, num_class_t=num_class_t, num_class_s=num_class)
    return test_accs[0]


def calculate_dataset_difficulty(source_dataset, source_modality, target_dataset, target_modality, unimodal_dict, 
                                device="cuda:0", first_finetune=False, remove=False, spath=None, tpath=None):
    _, _, test_s, _, source_path, embed_size, num_class = unimodal_dict[source_dataset]

    trains_t, valids_t, test_t, modalities_t, target_path, embed_size_t, num_class_t = unimodal_dict[target_dataset]

    model_path = source_path + '_' + source_modality + '_embed_flatten.pt'
    tmodel_path = target_path + '_' + str(target_modality) + '_embed_flatten.pt'

    if spath != None:
        model_path = spath
    if tpath != None:
        tmodel_path = tpath

    save_null_path = source_path + source_modality + str(target_modality) + '_null.pt'
    save_ygx_path =  source_path + source_modality + str(target_modality) + '_ygx.pt'

    model = torch.load(model_path).to(device)
    #classifier_target = torch.load(tmodel_path).to(device)
    unsqueezing = target_dataset == 'mimic'
    is_push = target_dataset == 'gentle_push'
    num_passes = len(model.to_logitslist)
    do_average = False

    #finetune null model and normal model on target dataset
    epochs = 80
    lr = 0.0008
    weight_decay = 0.001

    if len(target_modality) == 2:
        embed_size = 128
    elif len(target_modality) == 3:
        embed_size = 384

    all_trains, all_valids, all_tests, all_modals, single_modals = [], [], [], [], []
    for i in range(num_passes):
        all_trains.append(trains_t)
        all_valids.append(valids_t)
        all_tests.append(test_t)
        all_modals.append(modalities_t)
        if isinstance(target_modality, list):
            single_modals.append(target_modality)
        else:
            single_modals.append([target_modality])

    from private_test_scripts.perceivers.train_structure_multitask_tasksim import train

    if first_finetune:
        f_epochs = 40
        f_lr = 0.0001
        f_weight_decay = 0
        train(model, f_epochs, [trains_t], [valids_t], [test_t], [modalities_t], 'private_test_scripts/perceivers/finetuned.pt', lr=f_lr, device=device, train_weights=[1.0],
                is_affect=[False], unsqueezing=[unsqueezing], transpose=[False], evalweights=[1], start_from=0, weight_decay=f_weight_decay, 
                single_modality=[[target_modality]], transfer_type='finetune', embed_size=embed_size, num_class_t=num_class_t, num_class_s=num_class)

        model = torch.load('private_test_scripts/perceivers/finetuned.pt').to(device)
        model_path = 'private_test_scripts/perceivers/finetuned.pt'

    if single_modals == [['static', 'timeseries']] or single_modals == [['colorlessimage', 'audiospec']]:
        single_modals = None
    else:
        single_modals = single_modals[0]
    print(single_modals)

    _, entropy_null = train(model, epochs, all_trains, all_valids, all_tests, all_modals, save_null_path, lr=lr, device=device, train_weights=([1.0]*num_passes),
                is_affect=([False]*num_passes), unsqueezing=([unsqueezing]*num_passes), is_push=[is_push], transpose=([False]*num_passes), evalweights=([1]*num_passes), start_from=0, weight_decay=weight_decay, 
                single_modality=single_modals, transfer_type='finetune', embed_size=embed_size, num_class_t=num_class_t, num_class_s=num_class,
                null_model=True, calc_entropy=True, do_average=do_average, null_pvi=True)

    model = torch.load(model_path).to(device)

    if True: #source_dataset == target_dataset and source_modality == target_modality and spath == None:
        epochs = 0
        save_ygx_path = model_path


    _, entropy_ygx = train(model, epochs, all_trains, all_valids, all_tests, all_modals, save_ygx_path, lr=lr, device=device, train_weights=([1.0]*num_passes),
                is_affect=([False]*num_passes), unsqueezing=([unsqueezing]*num_passes), is_push=[is_push], transpose=([False]*num_passes), evalweights=([1]*num_passes), start_from=0, weight_decay=weight_decay, 
                single_modality=single_modals, transfer_type='finetune', embed_size=embed_size, num_class_t=num_class_t, num_class_s=num_class,
                null_model=False, calc_entropy=True, do_average=do_average)
    
    if remove and save_ygx_path != model_path:
        os.remove(save_null_path)
        os.remove(save_ygx_path)

    pvi_avgs=  []
    for i in range(len(entropy_null)):
        pvi_avg = np.mean(np.array(entropy_null[i]) - np.array(entropy_ygx[i]))
        pvi_avgs.append(pvi_avg)


    if not do_average:
        return pvi_avgs

    unified_pvi =  np.mean(np.array(avg_null) - np.array(avg_ygx))

    if remove and save_ygx_path != model_path:
        os.remove(save_null_path)
        os.remove(save_ygx_path)


    return pvi_avgs, unified_pvi


def l2norm_feature(s_dataset, s_single_modality, t_dataset, t_single_modality, unimodal_dict, spath=None, tpath=None):
    smodel_path = '_' + s_dataset + '_' + str([s_single_modality]) + '.pt'
    tmodel_path = '_' + t_dataset + '_' + str([t_single_modality]) + '.pt'
    if spath != None:
        smodel_path = spath
    if tpath != None:
        tmodel_path = tpath
    smodel = torch.load(smodel_path).to(device)
    tmodel = torch.load(tmodel_path).to(device)

    trains_s, valids_s, test_s, modalities_s, _, _, _ = unimodal_dict[s_dataset]
    trains_t, valids_t, test_t, modalities_t, _, _, _ = unimodal_dict[t_dataset]
    sunsqueezing = s_dataset == 'mimic'

    from private_test_scripts.perceivers.train_structure_multitask_tasksim import train
    feature_s = train(smodel, 0, [trains_s], [valids_s], [test_s], [modalities_s], smodel_path, lr=0, device=device, train_weights=[1.0],
                is_affect=[False], unsqueezing=[sunsqueezing], transpose=[False], evalweights=[1], start_from=0, weight_decay=0, 
                single_modality=s_single_modality, get_features=True)
    feature_t = train(tmodel, 0, [trains_t], [valids_t], [test_s], [modalities_s], tmodel_path, lr=0, device=device, train_weights=[1.0],
                is_affect=[False], unsqueezing=[sunsqueezing], transpose=[False], evalweights=[1], start_from=0, weight_decay=0, 
                single_modality=s_single_modality, get_features=True)


    fs = feature_s.flatten(start_dim=1)
    ft = feature_t.flatten(start_dim=1)

    m = torch.sum((fs - ft)**2, dim=1)
    return torch.sum(torch.sqrt(torch.sum((fs - ft)**2, dim=1)))

if __name__ == '__main__':
    '''
    # Compute metric using l2norm of each modality 
    accs = []
    for dataset in unimodal_dict:
        _, _, _, mod_type, _, _, _ = unimodal_dict[dataset]
        for single_modality in mod_type:
            taccs = []
            for tdataset in unimodal_dict:
                _, _, _, tmod_type, _, _, _ = unimodal_dict[tdataset]
                for tmod in tmod_type:
                    #if single_modality == tmod and dataset == tdataset:
                        #continue
                    print('s_dataset {} smod {} tdataset {} tmod {}'.format(dataset, single_modality, tdataset, tmod))
                    feature_dist = l2norm_feature(dataset, single_modality, tdataset, tmod, unimodal_dict)
                    print('feature_dist: {}'.format(feature_dist))
                    taccs.append(feature_dist)
                    accs.append(feature_dist)
            print('Feature distance dataset {} modality {}'.format(dataset, single_modality))
            for acc in taccs:
                print('%4f' % acc, end=' ')
            print('\n')

    print('l2 unimodal Feature Distances')
    count = 0 
    for acc in accs:
        print ('%.4f' % acc, end = ' ')
        if (count + 1) % 10 == 0:
            print('\n')
        count += 1

    '''


    datasets_list = [ ['mimic'], ['avmnist'], 
                      ['affect'], ['affect'], ['affect'],
                      ['humor'], ['humor'], ['humor']]

    mods_list = [ [['static', 'timeseries']], 
                 [['colorlessimage', 'audiospec']],
                [['feature1', 'feature2']],
                [['feature1', 'feature3']],
                [['feature2', 'feature3']],
                [['feature4', 'feature5']],
                [['feature4', 'feature3']],
                [['feature5', 'feature3']]
           ]
    paths = ['private_test_scripts/perceivers/mimicstatim1.pt', 
            'private_test_scripts/perceivers/avmnistimau.pt', 
            'private_test_scripts/perceivers/affectf1f2.pt',
            'private_test_scripts/perceivers/affectf1f3.pt',
            'private_test_scripts/perceivers/affectf2f3.pt',
            'private_test_scripts/perceivers/humorf4f5.pt',
            'private_test_scripts/perceivers/humorf4f3.pt',
            'private_test_scripts/perceivers/humorf5f3.pt'
            ]
  
    # Train unimodal models for each modality
    for i, datasets in enumerate(datasets_list):
        mods = mods_list[i]
        save_path = paths[i]
        print(datasets, mods)
        train_unimodal_model(unimodal_dict, datasets, mods, savepath=save_path)

    # Compute Heterogeneity matrix for pairs of modalities

    accs = []
    for m, path in enumerate(paths):
        source = datasets_list[m]
        print(datasets_list[m], mods_list[m])
        caccs = []
        for i, datasets in enumerate(datasets_list):
            if paths[i] != path:
                continue
            print(datasets, mods_list[i])
            #feature_dist = l2norm_feature(source[0], mods_list[m][0], datasets[0], mods_list[i][0], unimodal_dict, spath=path, tpath=paths[i])
            pvi = calculate_dataset_difficulty(source[0], '', datasets[0], mods_list[i][0], unimodal_dict, spath=path)
            print('PVI on dataset {}: {}'.format(datasets, pvi))
            caccs.append(pvi[0])
            accs.append(pvi[0])
        for acc in caccs:
            print(acc, end=' ')
        print('\n')

    count = 0
    for acc in accs:
        print (acc, end = ' ')
        if ((count + 1) % 10 == 0):
            print('\n')
        count += 1


    
    # Train unimodal models and calculate Heterogeneity Matrix for each pair of individual modalities
    accs = []
    count = 0
    for sdataset in unimodal_dict:
        _, _, _, smods, _, _, _ = unimodal_dict[sdataset]
        for smod in smods:
            taccs = []
            for tdataset in unimodal_dict:
                _, _, _, tmods, _, _, _ = unimodal_dict[tdataset]
                for tmod in tmods:
                    print(sdataset, smod, tdataset, tmod)
                    pvi = calculate_dataset_difficulty(sdataset, smod, tdataset, tmod, unimodal_dict)
                    accs.append(pvi)
                    taccs.append(pvi)
                    print ('PVI: %.4f' % pvi)
                    count+=1
            print('Dataset Difficulty Avg(PVI) for modality ' + sdataset + ' ' + smod)
            for acc in taccs:
                print ('%.4f' % acc, end = ' ')

    print('Dataset Difficulty (Avg PVI)') 
    count = 0
    for acc in accs:
        print ('%.4f' % acc, end = ' ')
        if (count + 1) % 10 == 0:
            print('\n')
        count += 1
    
    
    '''
    for dataset in unimodal_dict:
        if dataset != 'enrico' and dataset != 'gentle_push':
            continue
        _, _, _, mod_type, spath, _, _ = unimodal_dict[dataset]
        for single_modality in mod_type:
            save_path = spath + '_' + single_modality + '_embed_flatten.pt'
            print('model on dataset {} and modality {}'.format(dataset, single_modality))
            train_unimodal_model(unimodal_dict, [dataset], [[single_modality]], savepath=save_path)

    
    accs = []
    for sdataset in unimodal_dict:
        _, _, _, smods, _, _, _ = unimodal_dict[sdataset]
        for smod in smods:
            taccs = []
            isnew = sdataset == 'enrico' or sdataset == 'gentle_push'
            for tdataset in unimodal_dict:
                if not isnew and (tdataset != 'enrico' and tdataset != 'gentle_push'):
                    continue
                _, _, _, tmods, _, _, _ = unimodal_dict[tdataset]
                for tmod in tmods:
                    print(sdataset, smod, tdataset, tmod)
                    pvi = calculate_dataset_difficulty(sdataset, smod, tdataset, tmod, unimodal_dict)
                    accs.append(pvi)
                    taccs.append(pvi)
                    print ('PVI: %.4f' % pvi)
                    count+=1
            print('Dataset Difficulty Avg(PVI) for modality ' + sdataset + ' ' + smod)
            for acc in taccs:
                print ('%.4f' % acc, end = ' ')

    print('Dataset Difficulty (Avg PVI)') 
    count = 0
    for acc in accs:
        print ('%.4f' % acc, end = ' ')
        if (count + 1) % 10 == 0:
            print('\n')
        count += 1

    '''


    '''
    
    accs = []
    for sdataset in unimodal_dict:
        _, _, _, smods, _, _, _ = unimodal_dict[sdataset]
        for smod in smods:
            for tdataset in unimodal_dict:
                _, _, _, tmods, _, _, _ = unimodal_dict[tdataset]
                for tmod in tmods:
                    print(sdataset, smod, tdataset, tmod)
                    acc = target_data_source_classifier(sdataset, smod, tdataset, tmod, unimodal_dict, 'finetune')
                    accs.append(acc)

    print('Empirical Transfer with Embed (finetune)') 
    count = 0
    for acc in accs:
        print ('%.4f' % acc, end = ' ')
        if (count + 1) % 10 == 0:
            print('\n')
        count += 1

    for dataset in unimodal_dict:
        _, _, _, mod_type, _, _, _ = unimodal_dict[dataset]
        for single_modality in mod_type:
            print('model on dataset {} and modality {}'.format(dataset, single_modality))
            #train_unimodal_model(unimodal_dict, dataset, single_modality)

    accs = []
    for sdataset in unimodal_dict:
        _, _, _, smods, _, _, _ = unimodal_dict[sdataset]
        for smod in smods:
            for tdataset in unimodal_dict:
                _, _, _, tmods, _, _, _ = unimodal_dict[tdataset]
                for tmod in tmods:
                    print(sdataset, smod, tdataset, tmod)
                    acc = target_data_source_classifier(sdataset, smod, tdataset, tmod, unimodal_dict, 'retrain_head')
                    accs.append(acc)

    print('Empirical Transfer with Embed (Retrain_head)') 
    count = 0
    for acc in accs:
        print ('%.4f' % acc, end = ' ')
        if (count + 1) % 10 == 0:
            print('\n')
        count += 1

   ''' 


    '''
    with torch.no_grad():
        rets=[[],[],[],[]]
        for ii in range(len(test)):
            model.to_logits=classifier_target.to_logitslist[ii]
            model.embed = fourmulti.embed
            print(model.embed)
            if switch_embed:
                model.embed = classifier_target.embed
            totals=0
            corrects=0
            trues=[]
            preds=[]
            preds_l = []
            probs = []
            for jj in test[ii]:
                j=jj
                if is_affect:
                    j=jj[0]
                    j.append((jj[3].squeeze(1) >= 0).long())

                #if ismmimdb:
                #    j[0]=j[0].transpose(1,2)
                indict={}
                for i in range(0,len(modalities[ii])):
                    if modalities[ii][i] != target_modality:
                        continue
                    if unsqueezing:
                        indict[modalities[ii][i]]=j[i].float().unsqueeze(-1).to(device)
                    elif transpose:
                        indict[modalities[ii][i]]=j[i].float().to(device).transpose(1,2)
                    else:
                        indict[modalities[ii][i]]=j[i].float().to(device)

                assert(target_modality in indict and len(indict) == 1)
                source_mode = source_modality if use_source else None
                out=model(indict, source_mode=source_mode)

                if getattentionmap:
                    rets[ii].append(model.attns)
                    #break                       
                if ismmimdb:
                    trues.append(j[-1])
                    preds.append(torch.sigmoid(out).round())
                    probs.append(out.tolist())
                else:
                    for i in range(len(out)):
                        if isinstance(criterion,torch.nn.CrossEntropyLoss):
                            preds=torch.argmax(out,dim=1)
                            trues.append(j[-1].long()[i].item())
                            preds_l.append(preds[i].item())
                            probs.append(torch.nn.functional.softmax(out[i]).tolist())
                            if preds[i].item()==j[-1].long()[i].item():
                                corrects += 1
                        else:
                            print(out[i].item(), j[-1][i].item())
                            if (out[i].item() >= 0) == j[-1].long()[i].item():
                                corrects += 1
                        totals += 1
            if ismmimdb:
                trues=torch.cat(trues,0)
                preds=torch.cat(preds,0)
                f1_micro = f1_score(true, pred, average="micro")
                f1_macro = f1_score(true, pred, average="macro")
                accs = f1_macro
                print("test f1_macro: "+str(f1_macro)+" f1_micro: "+str(f1_micro))
            else:
                acc=float(corrects)/totals*100
                #testaccs.append(acc)
                print("test acc "+ target_dataset + ", modality: " + target_modality + ": " + '%.4f'% acc)
    
            #target_trues.append(trues)
            #toapp = preds_l if preds_l != [] else preds
            #target_preds.append(toapp)

        print(len(trues), len(preds_l), len(probs))

        return trues, preds_l, probs, acc
        '''

            






