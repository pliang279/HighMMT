import torch


def train(model,epochs,trains,valid,test,modalities,savedir,lr=0.001,weight_decay=0.0, optimizer=torch.optim.Adam, criterions=[torch.nn.CrossEntropyLoss(),torch.nn.CrossEntropyLoss()],valid_criterions=None,test_criterions=None,is_classification=[True,True],unsqueezing=[True,True], device="cuda:0",train_weights=[1.0,1.0],is_affect=[False,False],transpose=[False,False],valid_weights=None,encoder=None,modality_instance_mapping=None):
    #for param in model.parameters():
    #    print(param)
    if valid_criterions is None:
        valid_criterions = criterions
    if test_criterions is None:
        test_criterions = valid_criterions
    '''
    optim = optimizer(model.parameters(),lr=lr,weight_decay=weight_decay)
    bestacc=-999999999
    for ep in range(epochs):
        totalloss=[]
        totals=[]
        fulltrains=[]
        for i in range(len(trains)):
            count=0
            totalloss.append(0.0)
            totals.append(0)
            for j in trains[i]:
                if count >= len(fulltrains):
                    fulltrains.append({})
                if is_affect[i]:
                    jj=j[0]
                    jj.append((j[3].squeeze(1)>=0).long())
                    fulltrains[count][str(i)]=jj
                else:
                    fulltrains[count][str(i)]=j
                count += 1
        fulltrains.reverse()
        for js in fulltrains:
            optim.zero_grad()
            losses=0.0
            for ii in js:
                model.to_logits=model.to_logitslist[int(ii)]
                indict={}
                for i in range(len(modalities[int(ii)])):
                    if unsqueezing[int(ii)]:
                        indict[modalities[int(ii)][i]]=js[ii][i].float().unsqueeze(-1).to(device)
                    elif transpose[int(ii)]:
                        indict[modalities[ii][i]]=j[i].float().to(device).transpose(1,2)
                    else:
                        indict[modalities[int(ii)][i]]=js[ii][i].float().to(device)
                for mod in indict:
                    indict[mod].requires_grad=True
                #print(indict['colorlessimage'].size())
                #if 'audiospec' in indict:
                    #print(indict['colorlessimage'].size())
                    #print(indict['audiospec'].size())
                if encoder is not None:
                    indict = encoder(indict)
                out=model(indict, modality_instance_mapping=modality_instance_mapping)
                if is_classification[int(ii)]:
                    loss=criterions[int(ii)](out,js[ii][-1].long().to(device))
                else:
                    loss=criterions[int(ii)](out,js[ii][-1].to(device))
                losses += loss*train_weights[int(ii)]
                total=len(js[ii][0])
                totals[int(ii)] += total
                totalloss[int(ii)] += loss.item()*total
            losses.backward()
            optim.step()
            #print("We're at "+str(totals))
        for ii in range(len(trains)):
            print("epoch "+str(ep)+" train loss dataset " +str(ii)+": "+str(totalloss[ii]/totals[ii]))
        with torch.no_grad():
            accs=0.0
            for ii in range(len(valid)):
                totalloss=0.0
                totals=0
                corrects=0
                for jj in valid[ii]:
                    j=jj
                    if is_affect[ii]:
                        j=jj[0]
                        j.append((jj[3].squeeze(1) >= 0).long())
                    model.to_logits=model.to_logitslist[ii]
                    indict={}
                    for i in range(len(modalities[ii])):
                        if unsqueezing[ii]:
                            indict[modalities[ii][i]]=j[i].float().unsqueeze(-1).to(device)
                        elif transpose[ii]:
                            indict[modalities[ii][i]]=j[i].float().to(device).transpose(1,2)
                        else:
                            indict[modalities[ii][i]]=j[i].float().to(device)
                    if encoder is not None:
                        indict = encoder(indict)
                    out=model(indict, modality_instance_mapping=modality_instance_mapping)
                    if is_classification[int(ii)]:
                        loss=valid_criterions[ii](out,j[-1].long().to(device))
                    else:
                        loss=valid_criterions[ii](out,j[-1].to(device))
                    totalloss += loss.item()*len(j[0])
                    if is_classification[int(ii)]:
                        preds=torch.argmax(out,dim=1)
                        for i in range(len(preds)):
                            if preds[i].item()==j[-1].long()[i].item():
                                corrects += 1
                            totals += 1
                    else:
                        totals += len(out)
                if is_classification[int(ii)]:
                    acc=float(corrects)/totals
                    if valid_weights is None:
                        accs += acc
                    else:
                        accs += acc * valid_weights[int(ii)]
                    print("epoch "+str(ep)+" valid loss dataset"+str(ii)+": "+str(totalloss/totals)+" acc: "+str(acc))
                else:
                    loss = totalloss/totals
                    if valid_weights is None:
                        accs -= loss
                    else:
                        accs -= loss * valid_weights[int(ii)]
                    print("epoch "+str(ep)+" valid loss dataset"+str(ii)+": "+str(totalloss/totals))
            if accs > bestacc:
                print("save best")
                bestacc=accs
                torch.save(model,savedir)
    '''
    model=torch.load(savedir).to(device)
    with torch.no_grad():
        for ii in range(len(test)):
            model.to_logits=model.to_logitslist[ii]
            totalloss=0.0
            totals=0
            corrects=0
            for jj in test[ii]:            
                j=jj
                if is_affect[ii]:
                    j=jj[0]
                    j.append((jj[3].squeeze(1) >= 0).long())
                indict={}
                for i in range(0,len(modalities[ii])):
                    if unsqueezing[ii]:
                        indict[modalities[ii][i]]=j[i].float().unsqueeze(-1).to(device)
                    elif transpose[ii]:
                        indict[modalities[ii][i]]=j[i].float().to(device).transpose(1,2)
                    else:
                        indict[modalities[ii][i]]=j[i].float().to(device)
                if encoder is not None:
                    indict = encoder(indict)
                out=model(indict, modality_instance_mapping)

                if is_classification[int(ii)]:
                    preds=torch.argmax(out,dim=1)
                    for i in range(len(preds)):
                        if preds[i].item()==j[-1].long()[i].item():
                            corrects += 1
                        totals += 1
                else:
                    loss=test_criterions[ii](out,j[-1].to(device))
                    totalloss += loss.item()*len(j[0])
                    totals += len(out)
            if is_classification[int(ii)]:
                acc=float(corrects)/totals
                print("test acc dataset "+str(ii)+": "+str(ii)+" "+str(acc))
            else:
                print("test loss dataset "+str(ii)+": "+str(totalloss/totals))









