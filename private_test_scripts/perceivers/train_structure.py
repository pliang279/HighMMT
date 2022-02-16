import torch


def train(model,epochs,trains,valid,test,modalities,savedir,lr=0.001, optimizer=torch.optim.Adam, criterion=torch.nn.CrossEntropyLoss(),unsqueezing=True, device="cuda:0",transposing=False):
    #for param in model.parameters():
    #    print(param)
    optim = optimizer(model.parameters(),lr=lr)
    bestacc=0.0
    for ep in range(epochs):
        totalloss=0.0
        totals=0
        for j in trains:
            optim.zero_grad()
            indict={}
            for i in range(len(modalities)):
                if unsqueezing:
                    indict[modalities[i]]=j[i].float().unsqueeze(-1).to(device)
                elif transposing:
                    indict[modalities[i]]=j[i].float().transpose(1,2).to(device)
                else:
                    indict[modalities[i]]=j[i].float().to(device)
            for mod in indict:
                indict[mod].requires_grad=True
            #print(indict['colorlessimage'].size())
            out=model(indict)
            loss=criterion(out,j[-1].long().to(device))
            loss.backward()
            optim.step()
            totalloss += loss.item()*len(j[0])
            totals += len(j[0])
            #print("We're at "+str(totals))
        print("epoch "+str(ep)+" train loss: "+str(totalloss/totals))
        with torch.no_grad():
            totalloss=0.0
            totals=0
            corrects=0
            for j in valid:
                indict={}
                for i in range(len(modalities)):
                    if unsqueezing:
                        indict[modalities[i]]=j[i].float().unsqueeze(-1).to(device)
                    elif transposing:
                        indict[modalities[i]]=j[i].float().transpose(1,2).to(device)
                    else:
                        indict[modalities[i]]=j[i].float().to(device)
                out=model(indict)
                loss=criterion(out,j[-1].long().to(device))
                totalloss += loss.item()*len(j[0])
                preds=torch.argmax(out,dim=1)
                for i in range(len(preds)):
                    if preds[i].item()==j[-1].long()[i].item():
                        corrects += 1
                    totals += 1
            acc=float(corrects)/totals
            print("epoch "+str(ep)+" valid loss: "+str(totalloss/totals)+" acc: "+str(acc))
            if acc > bestacc:
                print("save best")
                bestacc=acc
                torch.save(model,savedir)
    model=torch.load(savedir).to(device)
    with torch.no_grad():
        totals=0
        corrects=0
        for j in test:            
            indict={}
            for i in range(0,len(modalities)):
                if unsqueezing:
                    indict[modalities[i]]=j[i].float().unsqueeze(-1).to(device)
                elif transposing:
                    indict[modalities[i]]=j[i].float().transpose(1,2).to(device)
                else:
                    indict[modalities[i]]=j[i].float().to(device)
            out=model(indict)

            preds=torch.argmax(out,dim=1)
            for i in range(len(preds)):
                if preds[i].item()==j[-1].long()[i].item():
                    corrects += 1
                totals += 1
        acc=float(corrects)/totals
        print("test acc: "+str(acc))









