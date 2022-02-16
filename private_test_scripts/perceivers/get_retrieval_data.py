import torch
import numpy as np

def to4by4aud(inp):
    #batch = inp.shape[0]
    b=[]
    for img in [inp]:
        for i in range(7):
            for j in range(7):
                b.append(img[i*16:i*16+16,j*16:j*16+16].flatten())
    b = np.array(b).reshape(7,7,256)
    return b


def to4by4aud2(inp):
    #batch = inp.shape[0]
    b=[]
    for img in [inp]:
        for i in range(25):
            for j in range(25):
                b.append(img[i*4:i*4+16,j*4:j*4+16].flatten())
    b = np.array(b).reshape(25,25,256)
    return b

def to4by4img(inp):
    #batch = inp.shape[0]
    b=[]
    a=torch.FloatTensor(inp).transpose(0,2).numpy()
    for img in [a]:
        for i in range(8):
            for j in range(8):
                b.append(img[i*4:i*4+4,j*4:j*4+4,:].flatten())
    b = np.array(b).reshape(8,8,48)
    return b

from torch.utils.data import DataLoader
def get_dataloader(file='../esc50-cifar/fulldata.pt',batch_size=32):
    trainpairs,valpairs,testpairs=torch.load(file)
    for l in trainpairs+valpairs+testpairs:
        l[1] = to4by4aud2(l[1])
        l[0] = to4by4img(l[0].reshape(3,32,32))/255.0
    trains = DataLoader(trainpairs,shuffle=True,num_workers=8,batch_size=batch_size)
    vals = DataLoader(valpairs,shuffle=False,num_workers=8,batch_size=batch_size)
    tests = DataLoader(testpairs,shuffle=False,num_workers=8,batch_size=batch_size)
    return trains,vals,tests
    
def get_dataloader2(file='../esc50-cifar/fulldata.pt',batch_size=32):
    trainpairs,valpairs,testpairs=torch.load(file)
    for l in trainpairs+valpairs+testpairs:
        l[1] = l[1].reshape(1,112,112)
        l[0] = l[0].reshape(3,32,32)/255.0
    trains = DataLoader(trainpairs,shuffle=True,num_workers=8,batch_size=batch_size)
    vals = DataLoader(valpairs,shuffle=False,num_workers=8,batch_size=batch_size)
    tests = DataLoader(testpairs,shuffle=False,num_workers=8,batch_size=batch_size)
    return trains,vals,tests

def get_cifar(file='../esc50-cifar/cifaronly.pt',batch_size=32):
    trainpairs,valpairs,testpairs=torch.load(file)
    for l in trainpairs+valpairs+testpairs:
        l[0] = to4by4img(l[0].reshape(3,32,32))/255.0
    trains = DataLoader(trainpairs,shuffle=True,num_workers=8,batch_size=batch_size)
    vals = DataLoader(valpairs,shuffle=False,num_workers=8,batch_size=batch_size)
    tests = DataLoader(testpairs,shuffle=False,num_workers=8,batch_size=batch_size)
    return trains,vals,tests
   
def get_esc(file='../esc50-cifar/esconly.pt',batch_size=32,repeats=1,to4by4=True):
    trainpairs,valpairs,testpairs=torch.load(file)
    for l in trainpairs+valpairs+testpairs:       
        if to4by4:
            l[0] = to4by4aud2(l[0])
        else:
            l[0] = l[0].reshape(112,112,1)
    trains = DataLoader(trainpairs*repeats,shuffle=True,num_workers=8,batch_size=batch_size)
    vals = DataLoader(valpairs,shuffle=False,num_workers=8,batch_size=batch_size)
    tests = DataLoader(testpairs,shuffle=False,num_workers=8,batch_size=batch_size)
    return trains,vals,tests

import random
random.seed(10)
def makepairs(cifar,esc):
    pairs=[]
    for i in range(17):
        for j in cifar[i]:
            pairs.append([j,esc[i][wrongclass(len(esc[i]),-1)],1])
            w=wrongclass(17,i)
            pairs.append([j,esc[w][wrongclass(len(esc[w]),-1)],0])
    return pairs

def wrongclass(a,b):
    c=random.randrange(0,a)
    if c==b:
        return wrongclass(a,b)
    else:
        return c
	
def to17list(datas):
    li=[]
    for i in range(17):
        li.append([])
    for i in datas:
        li[i[1]].append(i[0])
    return li

def get_complex_data(batch_size=32):
    cifartrain,cifarval,cifartest=get_cifar()
    esctrain,escval,esctest=get_esc()
    trainpairs=makepairs(to17list(cifartrain.dataset),to17list(esctrain.dataset))
    valpairs=makepairs(to17list(cifarval.dataset),to17list(escval.dataset))
    testpairs=makepairs(to17list(cifartest.dataset),to17list(esctest.dataset))
    trains = DataLoader(trainpairs,shuffle=True,num_workers=0,batch_size=batch_size)
    vals = DataLoader(valpairs,shuffle=False,num_workers=0,batch_size=batch_size)
    tests = DataLoader(testpairs,shuffle=False,num_workers=0,batch_size=batch_size)
    return trains,vals,tests
	    
    
