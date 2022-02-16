import torch
import matplotlib.pyplot as plt


def train(model,epochs,trains,valid,test,modalities,savedir,lr=0.001,weight_decay=0.0, optimizer=torch.optim.Adam, criterion=torch.nn.CrossEntropyLoss(),unsqueezing=[True,True], device="cuda:0",train_weights=[1.0,1.0],is_affect=[False,False],transpose=[False,False]):
    #for param in model.parameters():
    #    print(param)
    #params=[p for p in model.to_logitslist[0].parameters()]
    #params.append(model.embed)
    optim = optimizer(model.to_logitslist[0].parameters(),lr=lr,weight_decay=weight_decay)
    #optimizer.param_groups.append({'embed': model.embed })
    bestacc=0.0
    trainlosses=[]
    vallosses=[]
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
                out=model(indict)
                loss=criterion(out,js[ii][-1].long().to(device))
                losses += loss*train_weights[int(ii)]
                total=len(js[ii][0])
                totals[int(ii)] += total
                totalloss[int(ii)] += loss.item()*total
            losses.backward()
            optim.step()
            #print("We're at "+str(totals))
        for ii in range(len(trains)):
            print("epoch "+str(ep)+" train loss dataset " +str(ii)+": "+str(totalloss[ii]/totals[ii]))
            trainlosses.append(totalloss[ii]/totals[ii])
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
                    out=model(indict)
                    loss=criterion(out,j[-1].long().to(device))
                    totalloss += loss.item()*len(j[0])
                    preds=torch.argmax(out,dim=1)
                    for i in range(len(preds)):
                        if preds[i].item()==j[-1].long()[i].item():
                            corrects += 1
                        totals += 1
                vallosses.append(totalloss/totals)
                acc=float(corrects)/totals
                accs += acc
                print("epoch "+str(ep)+" valid loss dataset"+str(ii)+": "+str(totalloss/totals)+" acc: "+str(acc))
            if accs > bestacc:
                print("save best")
                bestacc=accs
                torch.save(model,savedir)
    model=torch.load(savedir).to(device)
    with torch.no_grad():
        for ii in range(len(test)):
            model.to_logits=model.to_logitslist[ii]
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
                out=model(indict)

                preds=torch.argmax(out,dim=1)
                for i in range(len(preds)):
                    if preds[i].item()==j[-1].long()[i].item():
                        corrects += 1
                    totals += 1
            acc=float(corrects)/totals
            print("test acc dataset "+str(ii)+": "+str(ii)+" "+str(acc))


    plt.plot(trainlosses)
    plt.plot(vallosses)
    plt.savefig('a.png')







