import sys
import os
sys.path.insert(1,os.getcwd())

import pickle
from torch.utils.data import DataLoader

f=open('/home/pliang/yiwei/MultiBench/humor.pkl','rb')
data=pickle.load(f)

def getdata(traindata,shuf=False,rate=1,batch_size=32,repeat=1):
    traindatas = [[traindata['vision'][i],traindata['audio'][i],traindata['text'][i],traindata['labels'][i][0]] for i in range(int(len(traindata['vision'])*rate))]*repeat
    #if rate < 1:
    #    for _,_,_,i in traindatas:
    #        print(i)
    return DataLoader(traindatas, shuffle=shuf, num_workers=0, batch_size=batch_size)

def get_dataloader(rate=1,train_batch_size=32,repeat=1):
    return getdata(data['train'],True,rate=rate,batch_size=train_batch_size,repeat=repeat),getdata(data['valid']),getdata(data['test'])
