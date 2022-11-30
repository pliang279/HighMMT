import torch
import numpy as np
from eval_scripts.performance import f1_score
from tqdm import tqdm
import copy
def train(model,epochs,trains,valid,test,modalities,savedir, uni_share_map, cross_share_map, num_uni_gps, num_cross_gps, lr=0.001,weight_decay=0.0, optimizer=torch.optim.Adam, criterion=torch.nn.CrossEntropyLoss(),unsqueezing=[True,True], device="cuda:0",train_weights=[1.0,1.0],is_affect=[False,False],is_push=[False, False], transpose=[False,False],ismmimdb=False,evalweights=None,recon=False, recon_weight=1, recon_criterion=torch.nn.MSELoss(),flips=-1, classesnum=[2,10,2,2],start_from=0,getattentionmap=False, one_optim=False, debug=False, lrs=None, reg_weights=None, regularise=False):
    #for param in model.parameters():
    #    print(param)

    '''
    model -> list of models with 2 sets of parameters. Unimodal params and cross-modal params
    uni_share_map -> maps modalities to indexes into models. This indicates which model's perceiver encoder is being used that modality
    cross_share_map -> maps pairs of modalities to indexes into models. Indicates which model's cross attention is used for that pair
    num_uni_gps -> number of separate groups that share unimodal parameters
    num_cross_gps -> number of separate groups that share crossmodal params

    idea: forward computation is done by taking each modality looking up the corresponding modality perceiver encoder group it belongs to, 
          and encoding it. Then for each pair of modalities in the dataset, it looks up the corresponding cross attion group it belongs to
          and passing the list of latent representations of each modality to it, then passing it to the corresponding dataset specific
          to_logits function. (all dataset to_logits are stored in model[0])

          Backward pass is computed by summing the losses of each dataset and calling optimizer.step for each parameter groups optimizer
    '''
    uni_optims = []
    cross_optims = []
    cross_names = []
    if lrs == None:
        lrs = [lr]*len(model)

    for i in range(len(model)):
        params_uni = []
        params_cross = []
        for n, p in model[i].cross_layers.named_parameters():    
            params_cross.append(p)
            cross_names.append('cross_layers.' + n)
        #print(params_cross)
        for n, p in model[i].named_parameters():
            if n not in cross_names:
                params_uni.append(p)
        

        optim_uni = optimizer(params_uni, lr=lrs[i], weight_decay=weight_decay)
        optim_cross = optimizer(params_cross, lr=lrs[i], weight_decay=weight_decay)

        uni_optims.append(optim_uni)
        cross_optims.append(optim_cross)

    parameter_uni = []
    parameter_cross = []
    for i in range(len(model)):
        x_params = []
        #test_uni = []
        #test_x = []
        #test_all = []
        uni_param_names = []
        for n, p in model[i].cross_layers.named_parameters():
            x_params.append('cross_layers.' + n)
            parameter_cross.append(p)
            #test_x.append(p)
        for n, p in model[i].named_parameters():
            #test_all.append(p)
            if n not in x_params:
                #uni_param_names.append(n)
                parameter_uni.append(p)
                #test_uni.append(p)
       #assert(len(test_all) == len(test_x) + len(test_uni))
        #for n, p in model[i].named_parameters():
            #assert(n in x_params or n in uni_param_names)

        


    optim_all_uni = optimizer(parameter_uni,lr=lr,weight_decay=weight_decay)
    optim_all_cross = optimizer(parameter_cross,lr=lr,weight_decay=weight_decay)



    bestacc=0.0
    best_acc_uni, best_acc_cross = {}, {}
    best_uni_param, best_cross_param = {}, {}
    for i in range(num_uni_gps):
        best_acc_uni[i] = 0.0
        best_uni_param[i] = None
    for i in range(num_cross_gps):
        best_acc_cross[i] = 0.0
        best_cross_param[i] = None

    returnrecs=[]
    for ep in range(epochs):
        toreturnrecs=[]
        totalloss=[]
        totals=[]
        fulltrains=[]
        indivcorrects=[]
        for i in range(len(trains)):
            toreturnrecs.append([])
            count=0
            totalloss.append(0.0)
            totals.append(0)
            indivcorrects.append(0)    
            for j in trains[i]:
                #print('iter')
                if count >= len(fulltrains):
                    fulltrains.append({})
                if is_affect[i]:
                    jj=j[0]
                    if isinstance(criterion,torch.nn.CrossEntropyLoss):
                        jj.append((j[3].squeeze(1)>=0).long())
                    else:
                        jj.append(j[3])
                    fulltrains[count][str(i)]=jj
                #elif ismmimdb:
                #    jj=[j[0].transpose(1,2),j[1],j[2]]
                #    fulltrains[count][str(i)]=jj
                else:
                    #print("iter")
                    fulltrains[count][str(i)]=j
                if i == flips:
                    j[-1] = (j[-1] + 1) % classesnum[i]
                count += 1
        fulltrains.reverse()
        fulltrains=fulltrains[start_from:]

        for js in tqdm(fulltrains):
            if not one_optim:
                for optim in uni_optims:
                    optim.zero_grad()
                for optim in cross_optims:
                    optim.zero_grad()
            else:
                optim_all_uni.zero_grad()
                optim_all_cross.zero_grad()

            for i in range(len(model)):
                for p in model[i].cross_layers.parameters():
                    p.requires_grad = True

            losses=0.0
            for ii in js:
                if debug and modalities[int(ii)] == ['static', 'timeseries']: continue
                
                #print(ii)
                #model.to_logits=model.to_logitslist[int(ii)]
                indict={}
                mod_indicts = []
                for i in range(len(modalities[int(ii)])):
                    mod_indict = {}
                    if unsqueezing[int(ii)]:
                        indict[modalities[int(ii)][i]]=js[ii][i].float().unsqueeze(-1).to(device)
                        mod_indict[modalities[int(ii)][i]]=js[ii][i].float().unsqueeze(-1).to(device)
                    elif transpose[int(ii)]:
                        indict[modalities[ii][i]]=js[i].float().to(device).transpose(1,2)
                        mod_indict[modalities[int(ii)][i]]=js[i].float().to(device).transpose(1,2)
                    else:
                        indict[modalities[int(ii)][i]]=js[ii][i].float().to(device)
                        mod_indict[modalities[int(ii)][i]]=js[ii][i].float().to(device)
                    if 'image_' in modalities[int(ii)][i]:
                        indict[modalities[int(ii)][i]] = js[ii][i].float().permute(0, 2, 3, 1).to(device)
                    #elif 'audio' in modalities[int(ii)][i]:
                     #   indict[modalities[int(ii)][i]] = js[ii][i].float().permute(0, 2, 1).to(device)
                    elif 'trajectory' in modalities[int(ii)][i]:
                        bs, len_seq, a, b = js[ii][i].shape
                        js[ii][i] = js[ii][i].reshape(bs*len_seq,a,b)
                        indict[modalities[int(ii)][i]] = js[ii][i].float().unsqueeze(-1).to(device)
                    elif 'pose' in modalities[int(ii)][i]:
                        bs, len_seq, c = js[ii][i].shape
                        js[ii][i] = js[ii][i].reshape(bs*len_seq,c)
                        indict[modalities[int(ii)][i]] = js[ii][i].float().unsqueeze(-1).to(device)
                    elif 'sensor' in modalities[int(ii)][i]:
                        bs, len_seq, c = js[ii][i].shape
                        js[ii][i] = js[ii][i].reshape(bs*len_seq,c)
                        indict[modalities[int(ii)][i]] = js[ii][i].float().unsqueeze(-1).to(device)
                    elif 'control' in modalities[int(ii)][i]:
                        bs, len_seq, c = js[ii][i].shape
                        js[ii][i] = js[ii][i].reshape(bs*len_seq,c)
                        indict[modalities[int(ii)][i]] = js[ii][i].float().unsqueeze(-1).to(device)
                    
                    mod_indict[modalities[int(ii)][i]] = indict[modalities[int(ii)][i]] 
                    
                    mod_indicts.append(mod_indict)

                for mod in indict:
                    indict[mod].requires_grad=True
                #print(indict['colorlessimage'].size())
                #if 'audiospec' in indict:
                    #print(indict['colorlessimage'].size())
                    #print(indict['audiospec'].size())

                if 'trajectory' in modalities[int(ii)]:
                    bs, len_seq, c = js[ii][-1].shape
                    js[ii][-1] = js[ii][-1].reshape(bs*len_seq, c)
                
                if recon:
                    out,rec = model(indict,use_recon=True)
                    stuffs = []
                    for modal in indict:
                        stuffs.append(torch.mean(indict[modal], dim=1))
                    origs=torch.cat(stuffs,dim=1)
                    loss=criterion(out,js[ii][-1].long().to(device))+ recon_weight*recon_criterion(rec,origs)
                else:
                    #unimodal encoding
                    latentout = []
                    for i, mod in enumerate(indict):
                        mod_in = mod_indicts[i] #{mod: indict[mod]}
                        uni_group = uni_share_map[mod]
                        mod_latent= model[uni_group](mod_in, get_latent=True)
                        latentout.append(mod_latent)
                    
                    assert(len(modalities[int(ii)]) == len(latentout))
                    outs = []

                    for i in range(len(latentout)):
                        for j in range(len(latentout)):
                            if i == j: continue
                            modi = modalities[int(ii)][i]
                            modj = modalities[int(ii)][j]
                            if (modi, modj) in cross_share_map:
                                share_group = cross_share_map[(modi, modj)]
                            else:
                                share_group = cross_share_map[(modj, modi)]
                            count_p_share = share_group
                            outs.append(model[share_group]({}, get_pre_logits=True, latents=[latentout[i], latentout[j]]))
                    catted = torch.cat(outs, dim=1)
                    out = model[0].to_logitslist[int(ii)](catted)
                    
                    #out=model(indict)
                    if ismmimdb:
                        loss=criterion(out,js[ii][-1].float().to(device))
                    elif is_push[int(ii)]:
                        loss=recon_criterion(out,js[ii][-1].float().to(device))
                    else:
                        loss=criterion(out,js[ii][-1].long().to(device))
                losses += loss*train_weights[int(ii)]
                total=len(js[ii][0])
                totals[int(ii)] += total
                totalloss[int(ii)] += loss.item()*total
                if not is_push[int(ii)]:
                    for i in range(total):
                        if torch.argmax(out[i]).item() == js[ii][-1][i]:
                            indivcorrects[int(ii)] += 1

            losses.backward()
            
            ''' 
            tparamx = 0
            tparam = 0
            all_n = []
            print(count_p_share)
            for n,p in model[count_p_share].cross_layers.named_parameters():
                all_n.append('cross_layer.' + n)
                if p.grad != None:
                    tparamx += np.prod(p.size())
            for n, p in model[count_p_share].named_parameters():
                if p.grad != None and n not in all_n:
                    tparam+= np.prod(p.size())
                else:
                    print(n)
            print(tparam, tparamx)
            '''
        
            


            ##REGULARIZATION STEP
           
            if regularise:
                reg_losses = 0.0

                for i in range(len(model)):
                    for p in model[i].cross_layers.parameters():
                        p.requires_grad = False

                for ii in js:
                    all_uni_gps = set()
                    for mod in modalities[int(ii)]:
                        all_uni_gps.add(uni_share_map[mod])

                    for reg_uni_gp in range(num_uni_gps):
                        if reg_uni_gp in all_uni_gps: 
                            continue

                        indict={}
                        mod_indicts = []
                        for i in range(len(modalities[int(ii)])):
                            mod_indict = {}
                            if unsqueezing[int(ii)]:
                                indict[modalities[int(ii)][i]]=js[ii][i].float().unsqueeze(-1).to(device)
                                mod_indict[modalities[int(ii)][i]]=js[ii][i].float().unsqueeze(-1).to(device)
                            elif transpose[int(ii)]:
                                indict[modalities[ii][i]]=j[i].float().to(device).transpose(1,2)
                                mod_indict[modalities[ii][i]]=j[i].float().to(device).transpose(1,2)
                            else:
                                indict[modalities[int(ii)][i]]=js[ii][i].float().to(device)
                                mod_indict[modalities[int(ii)][i]]=js[ii][i].float().to(device)
                            if 'image_' in modalities[int(ii)][i]:
                                indict[modalities[int(ii)][i]] = js[ii][i].float().permute(0, 2, 3, 1).to(device)
                            #elif 'audio' in modalities[int(ii)][i]:
                             #   indict[modalities[int(ii)][i]] = js[ii][i].float().permute(0, 2, 1).to(device)
                            elif 'trajectory' in modalities[int(ii)][i]:
                                bs, len_seq, a, b = js[ii][i].shape
                                js[ii][i] = js[ii][i].reshape(bs*len_seq,a,b)
                                indict[modalities[int(ii)][i]] = js[ii][i].float().unsqueeze(-1).to(device)
                            elif 'pose' in modalities[int(ii)][i]:
                                bs, len_seq, c = js[ii][i].shape
                                js[ii][i] = js[ii][i].reshape(bs*len_seq,c)
                                indict[modalities[int(ii)][i]] = js[ii][i].float().unsqueeze(-1).to(device)
                            elif 'sensor' in modalities[int(ii)][i]:
                                bs, len_seq, c = js[ii][i].shape
                                js[ii][i] = js[ii][i].reshape(bs*len_seq,c)
                                indict[modalities[int(ii)][i]] = js[ii][i].float().unsqueeze(-1).to(device)
                            elif 'control' in modalities[int(ii)][i]:
                                bs, len_seq, c = js[ii][i].shape
                                js[ii][i] = js[ii][i].reshape(bs*len_seq,c)
                                indict[modalities[int(ii)][i]] = js[ii][i].float().unsqueeze(-1).to(device)
                    
                            mod_indicts.append(mod_indict)

                        for mod in indict:
                            indict[mod].requires_grad=True
                        #print(indict['colorlessimage'].size())
                        #if 'audiospec' in indict:
                            #print(indict['colorlessimage'].size())
                            #print(indict['audiospec'].size())
                        if recon:
                            out,rec = model(indict,use_recon=True)
                            stuffs = []
                            for modal in indict:
                                stuffs.append(torch.mean(indict[modal], dim=1))
                            origs=torch.cat(stuffs,dim=1)
                            loss=criterion(out,js[ii][-1].long().to(device))+ recon_weight*recon_criterion(rec,origs)
                        else:
                            #unimodal encoding
                            latentout = []
                            for i, mod in enumerate(indict):
                                mod_in = mod_indicts[i] #{mod: indict[mod]}
                                mod_latent= model[reg_uni_gp](mod_in, get_latent=True)
                                latentout.append(mod_latent)
                            
                            assert(len(modalities[int(ii)]) == len(latentout))
                            outs = []
                            for i in range(len(latentout)):
                                for j in range(len(latentout)):
                                    if i == j: continue
                                    modi = modalities[int(ii)][i]
                                    modj = modalities[int(ii)][j]
                                    if (modi, modj) in cross_share_map:
                                        share_group = cross_share_map[(modi, modj)]
                                    else:
                                        share_group = cross_share_map[(modj, modi)]
                                    outs.append(model[share_group]({}, get_pre_logits=True, latents=[latentout[i], latentout[j]]))
                            catted = torch.cat(outs, dim=1)
                            out = model[0].to_logitslist[int(ii)](catted)
                            
                            #out=model(indict)
                            if ismmimdb:
                                loss=criterion(out,js[ii][-1].float().to(device))
                            elif is_push[int(ii)]:
                                loss=recon_criterion(out,js[ii][-1].float().to(device))
                            else:
                                loss=criterion(out,js[ii][-1].long().to(device))

                        reg_losses += loss*reg_weights[reg_uni_gp]

                reg_losses.backward()
            
            '''
            for i in range(len(model)):
                count = 0 
                for p in model[i].cross_layers.parameters():
                    print(p.grad)
                    count += 1
                    if count > 3:
                        break
            '''

            if debug:
                cross_names_1 = []
                for n, p in model[1].cross_layers.named_parameters():
                    cross_names_1.append('cross_layers.' + n)
                for n, p in model[1].named_parameters():
                    if n in cross_names_1:
                        if p.grad != None:
                            print(n)
                            assert(False)
                    else:
                        if 'logits' not in n and p.grad == None:
                            print(n, p.grad)
                            assert(False)
                        #assert(p.grad == None)
            if not one_optim:
                for optim in uni_optims:
                    optim.step()
                for optim in cross_optims:
                    optim.step()
            else:
                optim_all_uni.step()
                optim_all_cross.step()
            #print("We're at "+str(totals))


        for ii in range(len(trains)):
            if is_push[int(ii)]:
                print("epoch "+str(ep)+" train mse dataset " +str(ii)+": "+str(totalloss[ii]/totals[ii]))
            else:
                acc = float(indivcorrects[ii])/totals[ii]
                print("epoch "+str(ep)+" train loss dataset " +str(ii)+": "+str(totalloss[ii]/totals[ii]) + " acc: " +str(acc))
                toreturnrecs[ii].append(acc)
        with torch.no_grad():
            accs=0.0
            acc_uni, acc_cross = {}, {}
            for i in range(num_uni_gps):
                acc_uni[i] = 0.0
            for i in range(num_cross_gps):
                acc_cross[i] = 0.0

            for ii in range(len(valid)):
                totalloss=0.0
                totals=0
                corrects=0
                trues=[]
                preds=[]
                uni_params = set()
                cross_params = set()
                for jj in valid[ii]:
                    j=jj
                    if is_affect[ii]:
                        j=jj[0]
                        if isinstance(criterion,torch.nn.CrossEntropyLoss):
                            j.append((jj[3].squeeze(1)>=0).long())
                        else:
                            j.append(jj[3])
                    #if ismmimdb:
                    #    j[0]=j[0].transpose(1,2)
                    #model.to_logits=model.to_logitslist[ii]
                    indict={}
                    mod_indicts = []
                    for i in range(len(modalities[ii])):
                        mod_indict = {}
                        if unsqueezing[ii]:
                            indict[modalities[ii][i]]=j[i].float().unsqueeze(-1).to(device)
                            mod_indict[modalities[ii][i]]=j[i].float().unsqueeze(-1).to(device)
                        elif transpose[ii]:
                            indict[modalities[ii][i]]=j[i].float().to(device).transpose(1,2)
                            mod_indict[modalities[ii][i]]=j[i].float().to(device).transpose(1,2)
                        else:
                            indict[modalities[ii][i]]=j[i].float().to(device)
                            mod_indict[modalities[ii][i]]=j[i].float().to(device)
                        if 'image_' in modalities[int(ii)][i]:
                            indict[modalities[int(ii)][i]] = j[i].float().permute(0, 2, 3, 1).to(device)
                        #elif 'audio' in modalities[int(ii)][i]:
                        #   indict[modalities[int(ii)][i]] = js[ii][i].float().permute(0, 2, 1).to(device)
                        elif 'trajectory' in modalities[ii][i]:
                            bs, len_seq, a, b = j[i].shape
                            j[i] = j[i].reshape(bs*len_seq,a,b)
                            indict[modalities[ii][i]] = j[i].float().unsqueeze(-1).to(device)
                        elif 'pose' in modalities[ii][i]:
                            bs, len_seq, c = j[i].shape
                            j[i] = j[i].reshape(bs*len_seq,c)
                            indict[modalities[ii][i]] = j[i].float().unsqueeze(-1).to(device)
                        elif 'sensor' in modalities[ii][i]:
                            bs, len_seq, c = j[i].shape
                            j[i] = j[i].reshape(bs*len_seq,c)
                            indict[modalities[ii][i]] = j[i].float().unsqueeze(-1).to(device)
                        elif 'control' in modalities[ii][i]:
                            bs, len_seq, c = j[i].shape
                            j[i] = j[i].reshape(bs*len_seq,c)
                            indict[modalities[ii][i]] = j[i].float().unsqueeze(-1).to(device)
                        
                        mod_indict[modalities[ii][i]] = indict[modalities[ii][i]]

                        mod_indicts.append(mod_indict)

                    if 'trajectory' in modalities[ii]:
                        bs, len_seq, c = j[-1].shape
                        j[-1] = j[-1].reshape(bs*len_seq, c)

                    #unimodal encoding
                    latentout = []
                    for i, mod in enumerate(indict):
                        mod_in = mod_indicts[i] #{mod: indict[mod]}
                        uni_group = uni_share_map[mod]
                        uni_params.add(uni_group)
                        #print('adding {} to uni_params'.format(uni_group))
                        mod_latent= model[uni_group](mod_in, get_latent=True)
                        latentout.append(mod_latent)
                    
                    assert(len(modalities[ii]) == len(latentout))

                    #crossmodal attention
                    outs = []
                    for i in range(len(latentout)):
                        for k in range(len(latentout)):
                            if i == k: continue
                            modi = modalities[ii][i]
                            modj = modalities[ii][k]
                            if (modi, modj) in cross_share_map:
                                share_group = cross_share_map[(modi, modj)]
                            else:
                                share_group = cross_share_map[(modj, modi)]
                            cross_params.add(share_group)
                            outs.append(model[share_group]({}, get_pre_logits=True, latents=[latentout[i], latentout[k]]))
                    catted = torch.cat(outs, dim=1)
                    out = model[0].to_logitslist[ii](catted)

                    #out=model(indict)
                    if ismmimdb:
                        loss=criterion(out,j[-1].float().to(device))
                    elif is_push[int(ii)]:
                        loss=recon_criterion(out,j[-1].float().to(device))
                    else:
                        loss=criterion(out,j[-1].long().to(device))
                    totalloss += loss.item()*len(j[0])
                    if ismmimdb:
                        trues.append(j[-1])
                        preds.append(torch.sigmoid(out).round())
                        totals += len(j[-1])
                    elif is_push[ii]:
                        trues.append(j[-1])
                        preds.append(out)
                    else:
                        #print(preds)
                        for i in range(len(out)):
                            #print("we are here")
                            if isinstance(criterion,torch.nn.CrossEntropyLoss):
                                preds=torch.argmax(out,dim=1)
                                if preds[i].item()==j[-1].long()[i].item():
                                    corrects += 1
                            else:
                                if (out[i].item() >= 0) == (j[-1].long()[i].item() >= 0):
                                    corrects += 1
                            totals += 1
                if ismmimdb:
                    true=torch.cat(trues,0).to(device)
                    pred=torch.cat(preds,0)
                    f1_micro = f1_score(true, pred, average="micro")
                    f1_macro = f1_score(true, pred, average="macro")
                    accs = f1_macro
                    print("epoch "+str(ep)+" valid loss dataset"+str(ii)+": "+str(totalloss/totals)+" f1_macro: "+str(f1_macro)+" f1_micro: "+str(f1_micro))
                elif is_push[ii]:
                    true=torch.cat(trues,0).to(device)
                    pred=torch.cat(preds,0).to(device)
                    mse = torch.sqrt(recon_criterion(pred, true))
                    rmse_avg = (mse*0.0572766 + mse*0.06118315)/2
                    print("epoch " + str(ep)+ " valid mse loss dataset" + str(ii)+": " + str(rmse_avg.item()))
                    if evalweights is None:
                        accs -= rmse_avg
                    else:
                        for uni_group in uni_params:
                            #print('uni group: {}'.format(uni_group))
                            acc_uni[uni_group] -= rmse_avg*evalweights[ii]
                        for share_group in cross_params:
                            acc_cross[share_group] -= rmse_avg*evalweights[ii]
                        accs -= rmse_avg*evalweights[ii]
                else:
                    acc=float(corrects)/totals
                    if evalweights is None:
                        accs += acc
                    else:
                        #print(uni_params)
                        for uni_group in uni_params:
                            #print('uni group: {}'.format(uni_group))
                            acc_uni[uni_group] += acc*evalweights[ii]
                        for share_group in cross_params:
                            acc_cross[share_group] += acc*evalweights[ii]
                        accs += acc*evalweights[ii]
                    print("epoch "+str(ep)+" valid loss dataset"+str(ii)+": "+str(totalloss/totals)+" acc: "+str(acc))
                    toreturnrecs[ii].append(acc)
            
            for uni_group in acc_uni:
                #print('uni group valid: {}, acc: {}'.format(uni_group, acc_uni[uni_group]))
                if acc_uni[uni_group] > best_acc_uni[uni_group]:
                    best_uni_param[uni_group] =  copy.deepcopy(model[uni_group].state_dict())
                    best_acc_uni[uni_group] = acc_uni[uni_group]
            for share_group in acc_cross:
                if acc_cross[share_group] > best_acc_cross[share_group]:
                    best_cross_param[share_group] = copy.deepcopy(model[share_group].cross_layers.state_dict())
                    best_acc_cross[share_group] = acc_cross[share_group]
            '''
            if accs > bestacc:
                print("save best")
                bestacc=accs
                torch.save(model,savedir)
            returnrecs.append(toreturnrecs)
    model=torch.load(savedir).to(device)
    '''
    for uni_group in best_uni_param:
        #print('uni group load: {}'.format(uni_group))
        if best_uni_param[uni_group] != None:
            model[uni_group].load_state_dict(best_uni_param[uni_group])
    for share_group in best_cross_param:
        if best_cross_param[share_group] != None:
            model[share_group].cross_layers.load_state_dict(best_cross_param[share_group])
    testaccs=[]
    with torch.no_grad():
        rets=[[],[],[],[]]
        for ii in range(len(test)):
            #model.to_logits=model.to_logitslist[ii]
            totals=0
            corrects=0
            trues=[]
            preds=[]
            for jj in test[ii]:            
                j=jj
                if is_affect[ii]:
                    j=jj[0]
                    j.append((jj[3].squeeze(1) >= 0).long())

                #if ismmimdb:
                #    j[0]=j[0].transpose(1,2)
                indict={}
                mod_indicts=  []
                for i in range(0,len(modalities[ii])):
                    mod_indict = {}
                    if unsqueezing[ii]:
                        indict[modalities[ii][i]]=j[i].float().unsqueeze(-1).to(device)
                        mod_indict[modalities[ii][i]]=j[i].float().unsqueeze(-1).to(device)
                    elif transpose[ii]:
                        indict[modalities[ii][i]]=j[i].float().to(device).transpose(1,2)
                        mod_indict[modalities[ii][i]]=j[i].float().to(device).transpose(1,2)
                    else:
                        indict[modalities[ii][i]]=j[i].float().to(device)
                        mod_indict[modalities[ii][i]]=j[i].float().to(device)
                    if 'image_' in modalities[int(ii)][i]:
                        indict[modalities[int(ii)][i]] = j[i].float().permute(0, 2, 3, 1).to(device)
                    #elif 'audio' in modalities[int(ii)][i]:
                    #   indict[modalities[int(ii)][i]] = js[ii][i].float().permute(0, 2, 1).to(device)
                    elif 'trajectory' in modalities[ii][i]:
                        bs, len_seq, a, b = j[i].shape
                        j[i] = j[i].reshape(bs*len_seq,a,b)
                        indict[modalities[ii][i]] = j[i].float().unsqueeze(-1).to(device)
                    elif 'pose' in modalities[ii][i]:
                        bs, len_seq, c = j[i].shape
                        j[i] = j[i].reshape(bs*len_seq,c)
                        indict[modalities[ii][i]] = j[i].float().unsqueeze(-1).to(device)
                    elif 'sensor' in modalities[ii][i]:
                        bs, len_seq, c = j[i].shape
                        j[i] = j[i].reshape(bs*len_seq,c)
                        indict[modalities[ii][i]] = j[i].float().unsqueeze(-1).to(device)
                    elif 'control' in modalities[ii][i]:
                        bs, len_seq, c = j[i].shape
                        j[i] = j[i].reshape(bs*len_seq,c)
                        indict[modalities[ii][i]] = j[i].float().unsqueeze(-1).to(device)
                        
                    mod_indict[modalities[ii][i]] = indict[modalities[ii][i]]

                    mod_indicts.append(mod_indict)
                
                if 'trajectory' in modalities[ii]:
                    bs, len_seq, c = j[-1].shape
                    j[-1] = j[-1].reshape(bs*len_seq, c)

                #unimodal encoding
                latentout = []
                for i, mod in enumerate(indict):
                    mod_in = mod_indicts[i] #{mod: indict[mod]}
                    uni_group = uni_share_map[mod]
                    mod_latent= model[uni_group](mod_in, get_latent=True)
                    latentout.append(mod_latent)
                    
                assert(len(modalities[int(ii)]) == len(latentout))
                outs = []
                for i in range(len(latentout)):
                    for k in range(len(latentout)):
                        if i == k: continue
                        modi = modalities[ii][i]
                        modj = modalities[ii][k]
                        if (modi, modj) in cross_share_map:
                            share_group = cross_share_map[(modi, modj)]
                        else:
                            share_group = cross_share_map[(modj, modi)]
                        outs.append(model[share_group]({}, get_pre_logits=True, latents=[latentout[i], latentout[k]]))
                catted = torch.cat(outs, dim=1)
                out = model[0].to_logitslist[ii](catted)

                if getattentionmap:
                    rets[ii].append(model.attns)
                    #break                       
                if ismmimdb:
                    trues.append(j[-1])
                    preds.append(torch.sigmoid(out).round())
                elif is_push[int(ii)]:
                    trues.append(j[-1])
                    preds.append(out)
                else:
                    for i in range(len(out)):
                        if isinstance(criterion,torch.nn.CrossEntropyLoss):
                            preds=torch.argmax(out,dim=1)
                            if preds[i].item()==j[-1].long()[i].item():
                                corrects += 1
                        else:
                            print(out[i].item(), j[-1][i].item())
                            if (out[i].item() >= 0) == j[-1].long()[i].item():
                                corrects += 1
                        totals += 1

            if ismmimdb:
                true=torch.cat(trues,0).to(device)
                pred=torch.cat(preds,0).to(device)
                f1_micro = f1_score(true, pred, average="micro")
                f1_macro = f1_score(true, pred, average="macro")
                #accs = f1_macro
                print("test f1_macro: "+str(f1_macro)+" f1_micro: "+str(f1_micro))
            elif is_push[ii]:
                true=torch.cat(trues,0)
                pred=torch.cat(preds,0)
                mse = torch.sqrt(recon_criterion(pred, true))
                rmse_avg = (mse*0.0572766 + mse*0.06118315)/2
                print("test mse dataset "+str(ii)+": "+str(ii)+" "+str(rmse_avg.item()))
            elif not getattentionmap:
                acc=float(corrects)/totals
                testaccs.append(acc)
                print("test acc dataset "+str(ii)+": "+str(ii)+" "+str(acc))
    if getattentionmap:
        return rets
    return testaccs





def get_grads(test,model,modalities,device,unsqueezing,is_affect,transpose,mattersii):
    optimizer=torch.optim.SGD(model.parameters(),lr=0.0)
    for ii in range(len(test)):
        if ii != mattersii:
            continue
        model.to_logits=model.to_logitslist[ii]
        encoder_grads={}
        cross_grads={}
        count=0
        for jj in test[ii]: 
            count += 1
            optimizer.zero_grad()
            j=jj
            if is_affect[ii]:
                j=jj[0]
                j.append((jj[3].squeeze(1) >= 0).long())
            #if ismmimdb:
            #    j[0]=j[0].transpose(1,2
            indict={}
            for i in range(0,len(modalities[ii])):
                if unsqueezing[ii]:
                    indict[modalities[ii][i]]=j[i].float().unsqueeze(-1).to(device)
                elif transpose[ii]:
                    indict[modalities[ii][i]]=j[i].float().to(device).transpose(1,2)
                else:
                    indict[modalities[ii][i]]=j[i].float().to(device)
            out=model(indict)
            out = torch.nn.functional.softmax(out,dim=1)
            correctlabels=[out[i][j[-1][i].long().item()] for i in range(len(j[0]))]
            tograd=torch.mean(torch.stack(correctlabels))
            tograd.backward()
            for idx,param in enumerate(model.layers.parameters()):
                if str(idx) not in encoder_grads:
                    encoder_grads[str(idx)]=0.0
                encoder_grads[str(idx)] += torch.abs(param.grad.data)

            for idx,param in enumerate(model.cross_layers.parameters()):
                if str(idx) not in cross_grads:
                    cross_grads[str(idx)]=0.0
                cross_grads[str(idx)] += param.grad.data

        for idx in encoder_grads:
            encoder_grads[idx] /= count
        for idx in cross_grads:
            cross_grads[idx] /= count

    return encoder_grads,cross_grads



