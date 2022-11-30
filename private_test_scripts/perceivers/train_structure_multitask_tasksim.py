import torch
import numpy as np
from eval_scripts.performance import f1_score
from tqdm import tqdm
def train(model,epochs,trains,valid,test,modalities,savedir,lr=0.001,weight_decay=0.0, optimizer=torch.optim.Adam, criterion=torch.nn.CrossEntropyLoss(),unsqueezing=[True,True], device="cuda:0",train_weights=[1.0,1.0],is_affect=[False,False],is_push=[False,False], transpose=[False,False],ismmimdb=False,evalweights=None,recon=False, recon_weight=1, recon_criterion=torch.nn.MSELoss(),flips=-1, classesnum=[2,10,2,2],start_from=0,getattentionmap=False, single_modality=None, transfer_type='', embed_size=0, num_class_t=0, num_class_s=0, null_model=False, calc_entropy=False, pvi_loss=False, pvi_models=[], reuse_tologits=False, do_average=False,
null_pvi=False, get_features=False):
    #for param in model.parameters():
    #    print(param)
    finetune, retrain_head = False, False
    if transfer_type == 'finetune':
        finetune = True
        #model.to_logits = model.to_logitslist[0]
    elif transfer_type == 'retrain_head':
        retrain_head = True
        #model.to_logits = model.to_logitslist[0]
    elif pvi_loss:
        model.to_logits = model.to_logitslist[0]
        null_model, ygx_model = torch.load(pvi_models[0]).to(device), torch.load(pvi_models[1]).to(device)
        for p in null_model.parameters():
            p.requires_grad=False
        for p in ygx_model.parameters():
            p.requires_grad=False

    if finetune and not calc_entropy:
        for p in model.parameters():
            p.requires_grad=False
        for p in model.to_logitslist.parameters():
            p.requires_grad=True
        #model.embed.requires_grad=True
        parameters = []
        for p in model.to_logitslist.parameters():
            parameters.append(p)
    elif retrain_head:
        for p in model.parameters():
            p.requires_grad=False
        for p in model.to_logitslist.parameters():
            p.requires_grad=True
        parameters = model.to_logitslist.parameters()
    elif pvi_loss:
        for p in model.to_logits.parameters():
            p.requires_grad=False
        parameters = [p for p in model.parameters() if p not in model.to_logits.parameters()]
    else:
        parameters = model.parameters()

    #for param in model.parameters():
        #if param not in parameters:
            #param.requires_grad=False
        #print(param, param.requires_grad)

    optim = optimizer(parameters,lr=lr,weight_decay=weight_decay)
    bestacc=0.0
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
            optim.zero_grad()
            losses=0.0
            for ii in js:
                #print(ii)
                if reuse_tologits:
                    model.to_logits = model.to_logitslist[0]
                else:
                    model.to_logits=model.to_logitslist[int(ii)]

                if finetune and num_class_t != num_class_s:
                    model.to_logits[1] = torch.nn.Linear(embed_size,num_class_t).to(device)
                
                indict={}
                for i in range(len(modalities[int(ii)])):
                    if single_modality != None and (modalities[int(ii)][i] not in single_modality):
                        continue
                    if unsqueezing[int(ii)]:
                        indict[modalities[int(ii)][i]]=js[ii][i].float().unsqueeze(-1).to(device)
                    elif transpose[int(ii)]:
                        indict[modalities[ii][i]]=j[i].float().to(device).transpose(1,2)
                    else:
                        indict[modalities[int(ii)][i]]=js[ii][i].float().to(device)
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
                    
                    if null_model:
                        indict[modalities[int(ii)][i]] = torch.zeros(indict[modalities[int(ii)][i]].shape).to(device)

                                   
                # print('number of modalities ZEROED: {} OUT OF {}'.format(count, len(modalities[int(ii)])))

                if calc_entropy and null_model:
                    for mod in indict:
                        assert(torch.count_nonzero(indict[mod]).item() == 0)

                for mod in indict:
                    #print(mod)
                    indict[mod].requires_grad=True
                #print(indict['colorlessimage'].size())
                #if 'audiospec' in indicte
                    #print(indict['colorlessimage'].size())
                    #print(indict['audiospec'].size())
                #print('\n')

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
                elif pvi_loss:
                    out = model(indict, get_catted=True)
                    logits_null = null_model.to_logits(out)
                    logits_ygx = ygx_model.to_logits(out)
                    loss = 0.0
                    for i in range(len(logits_null)):
                        probsn = torch.nn.functional.softmax(logits_null)
                        label_probn = probsn[js[ii][-1].long()[i].item()].item()
                        hnull = torch.log2(label_probn)

                        probsygx = torch.nn.functional.softmax(logits_ygx)
                        label_probygx = probsygx[js[ii][-1].long()[i].item()].item()
                        hygx = torch.log2(label_probygx)
                        loss += (hnull - hygx)/len(js[ii][0])
                else:
                    out=model(indict, unimodal=(len(indict)==1), null_pvi=null_pvi)
                    if ismmimdb:
                        loss=criterion(out,js[ii][-1].float().to(device))
                    elif is_push[int(ii)]:
                        loss=torch.sqrt(recon_criterion(out, js[ii][-1].float().to(device)))
                    else:
                        loss=criterion(out,js[ii][-1].long().to(device))
                losses += loss*train_weights[int(ii)]
                total=len(js[ii][0])
                #print('total: {}'.format(total))
                totals[int(ii)] += total
                totalloss[int(ii)] += loss.item()*total
                if not is_push[int(ii)]:
                    for i in range(total):
                        if torch.argmax(out[i]).item() == js[ii][-1][i]:
                            indivcorrects[int(ii)] += 1
            losses.backward()
            optim.step()
            #print("We're at "+str(totals))
        for ii in range(len(trains)):
            if is_push[int(ii)]:
                if finetune or ((ep + 1) %5 == 0):
                    print("epoch "+str(ep)+" train mse loss dataset " +str(ii)+": "+str(totalloss[ii]/totals[ii]))
            else:
                acc = float(indivcorrects[ii])/totals[ii]
                if finetune or ((ep + 1) %5 == 0):
                    print("epoch "+str(ep)+" train loss dataset " +str(ii)+": "+str(totalloss[ii]/totals[ii]) + " acc: " +str(acc))
                toreturnrecs[ii].append(acc)
        with torch.no_grad():
            accs=0.0
            for ii in range(len(valid)):
                totalloss=0.0
                totals=0
                corrects=0
                trues=[]
                preds=[]
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
                    if reuse_tologits:
                        model.to_logits = model.to_logitslist[0]
                    else:
                        model.to_logits=model.to_logitslist[ii]
                    indict={}
                    for i in range(len(modalities[ii])):
                        if single_modality != None and (modalities[ii][i] not in single_modality):
                            continue
                        if unsqueezing[ii]:
                            indict[modalities[ii][i]]=j[i].float().unsqueeze(-1).to(device)
                        elif transpose[ii]:
                            indict[modalities[ii][i]]=j[i].float().to(device).transpose(1,2)
                        else:
                            indict[modalities[ii][i]]=j[i].float().to(device)
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

                        if null_model:
                            indict[modalities[ii][i]] = torch.zeros(indict[modalities[ii][i]].shape).to(device)

                    if 'trajectory' in modalities[int(ii)]:
                        bs, len_seq, c = j[-1].shape
                        j[-1] = j[-1].reshape(bs*len_seq, c)


                    out=model(indict, unimodal=(len(indict)==1), null_pvi=null_pvi)
                    if ismmimdb:
                        loss=criterion(out,j[-1].float().to(device))
                    elif is_push[ii]:
                        loss=recon_criterion(out, j[-1].float().to(device))
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
                    true=torch.cat(trues,0)
                    pred=torch.cat(preds,0)
                    f1_micro = f1_score(true, pred, average="micro")
                    f1_macro = f1_score(true, pred, average="macro")
                    accs = f1_macro
                    print("epoch "+str(ep)+" valid loss dataset"+str(ii)+": "+str(totalloss/totals)+" f1_macro: "+str(f1_macro)+" f1_micro: "+str(f1_micro))
                elif is_push[ii]:
                    true = torch.cat(trues, 0).to(device)
                    pred=torch.cat(preds, 0).to(device)
                    mse = torch.sqrt(recon_criterion(pred, true))
                    rmse_avg = (mse*0.0572766 + mse*0.06118315)/2
                    if finetune or ((ep + 1)%5 == 0):
                        print("epoch "+str(ep)+" valid mse loss dataset"+str(ii)+": "+str(rmse_avg.item()))
                    if evalweights is None:
                        accs -= rmse_avg
                    else:
                        accs -= rmse_avg*evalweights[ii]
                else:
                    acc=float(corrects)/totals
                    #print(corrects)
                    #print(totals)
                    if evalweights is None:
                        accs += acc
                    else:
                        accs += acc*evalweights[ii]
                    if finetune or ((ep + 1)%5 == 0):
                        print("epoch "+str(ep)+" valid loss dataset"+str(ii)+": "+str(totalloss/totals)+" acc: "+str(acc))
                    toreturnrecs[ii].append(acc)
            if accs >= bestacc:
                print("save best")
                bestacc=accs
                torch.save(model,savedir)
            returnrecs.append(toreturnrecs)
    model=torch.load(savedir).to(device)
    testaccs=[]
    with torch.no_grad():
        rets=[[],[],[],[]]
        avg_corrects = 0
        datotals = 0
        entropies = []
        avg_entropy = []
        for ii in range(len(test)):
            if reuse_tologits:
                model.to_logits = model.to_logitslist[0]
            else:
                model.to_logits=model.to_logitslist[ii]
            totals=0
            corrects=0
            trues=[]
            preds=[]
            entropy= [] 
            for jj in test[ii]:            
                j=jj
                if is_affect[ii]:
                    j=jj[0]
                    j.append((jj[3].squeeze(1) >= 0).long())

                #if ismmimdb:
                #    j[0]=j[0].transpose(1,2)
                indict={}
                for i in range(0,len(modalities[ii])):
                    if single_modality != None and (modalities[ii][i] not in single_modality):
                            continue
                    if unsqueezing[ii]:
                        indict[modalities[ii][i]]=j[i].float().unsqueeze(-1).to(device)
                    elif transpose[ii]:
                        indict[modalities[ii][i]]=j[i].float().to(device).transpose(1,2)
                    else:
                        indict[modalities[ii][i]]=j[i].float().to(device)
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
                        

                    if null_model:
                        indict[modalities[int(ii)][i]] = torch.zeros(indict[modalities[ii][i]].shape).to(device)
                
                if get_features:
                    latentout = model(indict, unimodal=(len(indict)==1), get_latent=True)
                    out = model(indict, unimodal=(len(indict)==1), null_pvi=null_pvi)
                    return latentout

                if 'trajectory' in modalities[int(ii)]:
                        bs, len_seq, c = j[-1].shape
                        j[-1] = j[-1].reshape(bs*len_seq, c)

                out=model(indict, unimodal=(len(indict)==1), null_pvi=null_pvi)

                if getattentionmap:
                    rets[ii].append(model.attns)
                    #break                       
                if ismmimdb:
                    trues.append(j[-1])
                    preds.append(torch.sigmoid(out).round())
                elif is_push[ii]:
                    trues.append(j[-1])
                    preds.append(out)
                else:
                    for i in range(len(out)):
                        if isinstance(criterion,torch.nn.CrossEntropyLoss):
                            preds=torch.argmax(out,dim=1)
                            if preds[i].item()==j[-1].long()[i].item():
                                corrects += 1

                            probs = torch.nn.functional.softmax(out[i])
                            label_prob = probs[j[-1].long()[i].item()].item()
                            entropy.append(-1 * np.log2(label_prob))

                        else:
                            print(out[i].item(), j[-1][i].item())
                            if (out[i].item() >= 0) == j[-1].long()[i].item():
                                corrects += 1
                        totals += 1
                    if do_average and ii == 0:
                        out_catted = model(indict, unimodal=(len(indict)==1), get_catted=True)
                        outs = []
                        num_passes = len(model.to_logitslist)
                        for i in range(num_passes):
                            outs.append(model.to_logitslist[i](out_catted))
                        
                        for i in range(len(outs[0])):
                            prob_i = torch.nn.functional.softmax(outs[0][i])
                            for passnum in range(1, num_passes):
                                prob_i += torch.nn.functional.softmax(outs[passnum][i])
                            prob_i /= float(num_passes)
                            label_prob_i = prob_i[j[-1].long()[i].item()].item()
                            avg_entropy.append(-1 * np.log2(label_prob_i))

                            if torch.argmax(prob_i).item() == j[-1].long()[i].item():
                                avg_corrects += 1
                            datotals += 1

                
            
            if ismmimdb:
                true=torch.cat(trues,0)
                pred=torch.cat(preds,0)
                f1_micro = f1_score(true, pred, average="micro")
                f1_macro = f1_score(true, pred, average="macro")
                #accs = f1_macro
                print("test f1_macro: "+str(f1_macro)+" f1_micro: "+str(f1_micro))
            elif is_push[ii]:
                true=torch.cat(trues,0).to(device)
                pred=torch.cat(preds, 0).to(device)
                mse = torch.sqrt(recon_criterion(pred, true))
                rmse_avg = (mse*0.0572766 + mse*0.06118315)/2
                print("test mse loss dataset "+str(ii)+": "+str(ii)+" "+str(rmse_avg.item()))
            elif not getattentionmap:
                acc=float(corrects)/totals
                testaccs.append(acc)
                if entropy != []:
                    entropies.append(entropy)
                print("test acc dataset "+str(ii)+": "+str(ii)+" "+str(acc))


    if do_average:
        print('acc on dataset using average of classifiers: {}'.format(avg_corrects/totals))
        return testaccs, entropies, avg_entropy
    if getattentionmap:
        return rets
    if calc_entropy:
        return testaccs, entropies
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



