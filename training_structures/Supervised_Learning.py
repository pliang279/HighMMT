
import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
import time
from eval_scripts.performance import AUPRC,f1_score,accuracy
from eval_scripts.complexity import all_in_one_train, all_in_one_test
from eval_scripts.robustness import relative_robustness, effective_robustness, single_plot
from tqdm import tqdm
#import pdb

softmax = nn.Softmax()

class MMDL(nn.Module):
    def __init__(self,encoders,fusion,head,has_padding=False):
        super(MMDL,self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.fuse = fusion
        self.head = head
        self.has_padding=has_padding
        self.fuseout = None
        self.reps = []    
    def forward(self,inputs,training=False):
        outs = []
        if self.has_padding:
            for i in range(len(inputs[0])):
                outs.append(self.encoders[i]([inputs[0][i],inputs[1][i]], training=training))
        else:
            for i in range(len(inputs)):
                outs.append(self.encoders[i](inputs[i], training=training))
        self.reps=outs
        out = self.fuse(outs, training=training)
       # print(out.size())
        self.fuseout = out
        if type(out) is tuple:
            out = out[0]
        return self.head(out, training=training)

def deal_with_objective(objective,pred,truth,args):
    if type(objective)==nn.CrossEntropyLoss:
        if len(truth.size())==len(pred.size()):
            truth1 = truth.squeeze(len(pred.size())-1)
        else:
            truth1 = truth
        return objective(pred,truth1.long().cuda())
    elif type(objective)==nn.MSELoss or type(objective)==nn.modules.loss.BCEWithLogitsLoss:
        return objective(pred,truth.float().cuda())
    else:
        return objective(pred,truth,args)

# encoders: list of modules, unimodal encoders for each input modality in the order of the modality input data.
# fusion: fusion module, takes in outputs of encoders in a list and outputs fused representation
# head: classification or prediction head, takes in output of fusion module and outputs the classification or prediction results that will be sent to the objective function for loss calculation
# total_epochs: maximum number of epochs to train
# additional_optimizing_modules: list of modules, include all modules that you want to be optimized by the optimizer other than those in encoders, fusion, head (for example, decoders in MVAE)
# is_packed: whether the input modalities are packed in one list or not (default is False, which means we expect input of [tensor(20xmodal1_size),(20xmodal2_size),(20xlabel_size)] for batch size 20 and 2 input modalities)
# early_stop: whether to stop early if valid performance does not improve over 7 epochs
# task: type of task, currently support "classification","regression","multilabel"
# optimtype: type of optimizer to use
# lr: learning rate
# weight_decay: weight decay of optimizer
# objective: objective function, which is either one of CrossEntropyLoss, MSELoss or BCEWithLogitsLoss or a custom objective function that takes in three arguments: prediction, ground truth, and an argument dictionary.
# auprc: whether to compute auprc score or not
# save: the name of the saved file for the model with current best validation performance
# validtime: whether to show valid time in seconds or not
# objective_args_dict: the argument dictionary to be passed into objective function. If not None, at every batch the dict's "reps", "fused", "inputs", "training" fields will be updated to the batch's encoder outputs, fusion module output, input tensors, and boolean of whether this is training or validation, respectively.
# input_to_float: whether to convert input to float type or not
# clip_val: grad clipping limit
# track_complexity: whether to track training complexity or not
def train(
    encoders,fusion,head,train_dataloader,valid_dataloader,total_epochs,additional_optimizing_modules=[],is_packed=False,
    early_stop=False,task="classification",optimtype=torch.optim.RMSprop,lr=0.001,weight_decay=0.0,
    objective=nn.CrossEntropyLoss(),auprc=False,save='best.pt',validtime=False, objective_args_dict=None,input_to_float=True,clip_val=8,
    track_complexity=True):
    model = MMDL(encoders,fusion,head,is_packed).cuda()
    def trainprocess():
        additional_params=[]
        for m in additional_optimizing_modules:
            additional_params.extend([p for p in m.parameters() if p.requires_grad])
        op = optimtype([p for p in model.parameters() if p.requires_grad]+additional_params,lr=lr,weight_decay=weight_decay)
        bestvalloss = 10000
        bestacc = 0
        bestf1 = 0
        patience = 0
    
        def processinput(inp):
            if input_to_float:
                return inp.float()
            else:
                return inp
        train_accs=[]
        valid_accs=[]
        for epoch in range(total_epochs):
            totalloss = 0.0
            totals = 0
            pred=[]
            true=[]
            model.train()
            for j in train_dataloader:
                op.zero_grad()
                if is_packed:
                    with torch.backends.cudnn.flags(enabled=False):
                        out=model([[processinput(i).cuda() for i in j[0]], j[1]],training=True)
                    
                else:
                    out=model([processinput(i).cuda() for i in j[:-1]],training=True)
                if not (objective_args_dict is None):
                    objective_args_dict['reps']=model.reps
                    objective_args_dict['fused']=model.fuseout
                    objective_args_dict['inputs']=j[:-1]
                    objective_args_dict['training']=True
                    objective_args_dict['model']=model
                pred.append(torch.argmax(out,1))
                true.append(j[-1])
                loss=deal_with_objective(objective,out,j[-1],objective_args_dict)
                totalloss += loss * len(j[-1])
                totals+=len(j[-1])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                op.step()
            pred = torch.cat(pred, 0)
            true = torch.cat(true, 0)
            acc=accuracy(true,pred)
            train_accs.append(acc)
            print("Epoch "+str(epoch)+" train loss: "+str(totalloss/totals)+" acc "+str(acc))
            validstarttime=time.time()
            if validtime:
                print("train total: "+str(totals))
            model.eval()
            with torch.no_grad():
                totalloss = 0.0
                pred = []
                true = []
                pts = []
                for j in valid_dataloader:
                    if is_packed:
                        out=model([[processinput(i).cuda() for i in j[0]], j[1]],training=False)
                    else:
                        out = model([processinput(i).cuda() for i in j[:-1]],training=False)

                    if not (objective_args_dict is None):
                        objective_args_dict['reps']=model.reps
                        objective_args_dict['fused']=model.fuseout
                        objective_args_dict['inputs']=j[:-1]
                        objective_args_dict['training']=False
                    loss=deal_with_objective(objective,out,j[-1],objective_args_dict)
                    totalloss += loss*len(j[-1])
                    #print(totalloss)
                    if task == "classification":
                        pred.append(torch.argmax(out, 1))
                    elif task == "multilabel":
                        pred.append(torch.sigmoid(out).round())
                    true.append(j[-1])
                    if auprc:
                        #pdb.set_trace()
                        sm=softmax(out)
                        pts += [(sm[i][1].item(), j[-1][i].item()) for i in range(j[-1].size(0))]
            if pred:
                pred = torch.cat(pred, 0)
            true = torch.cat(true, 0)
            totals = true.shape[0]
            valloss=totalloss/totals
            if task == "classification":
                acc = accuracy(true, pred)
                valid_accs.append(acc)
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss)+\
                    " acc: "+str(acc))
                if acc > bestacc:
                    patience = 0
                    bestacc = acc
                    print("Saving Best")
                    torch.save(model, save)
                else:
                    patience += 1
            elif task == "multilabel":
                f1_micro = f1_score(true, pred, average="micro")
                f1_macro = f1_score(true, pred, average="macro")
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss)+\
                    " f1_micro: "+str(f1_micro)+" f1_macro: "+str(f1_macro))
                if f1_macro>bestf1:
                    patience = 0
                    bestf1=f1_macro
                    print("Saving Best")
                    torch.save(model,save)
                else:
                    patience += 1
            elif task == "regression":
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss.item()))
                if valloss<bestvalloss:
                    patience = 0
                    bestvalloss=valloss
                    print("Saving Best")
                    torch.save(model,save)
                else:
                    patience += 1
            if early_stop and patience > 7:
                break
            if auprc:
                print("AUPRC: "+str(AUPRC(pts)))
            validendtime=time.time()
            if validtime:
                print("valid time:  "+str(validendtime-validstarttime))
                print("Valid total: "+str(totals))
        f=open("track.csv",'w+')
        for t,v in zip(train_accs,valid_accs):
            f.write(str(t)+','+str(v)+'\n')
        f.close()

    if track_complexity:
        all_in_one_train(trainprocess,[model]+additional_optimizing_modules)
    else:
        trainprocess()

def single_test(
    model,test_dataloader,is_packed=False,
    criterion=nn.CrossEntropyLoss(),task="classification",auprc=False,input_to_float=True):
    def processinput(inp):
        if input_to_float:
            return inp.float()
        else:
            return inp
    with torch.no_grad():
        totalloss = 0.0
        pred=[]
        true=[]
        pts=[]
        for j in test_dataloader:
            if is_packed:
                out=model([[processinput(i).cuda() for i in j[0]], j[1]],training=False)
            else:
                out = model([processinput(i).float().cuda() for i in j[:-1]],training=False)
            if type(criterion) == torch.nn.modules.loss.BCEWithLogitsLoss or type(criterion) == torch.nn.MSELoss:
                loss=criterion(out, j[-1].float().cuda())
            elif type(criterion) == torch.nn.CrossEntropyLoss:
                loss=criterion(out, j[-1].long().cuda())
            else:
                loss = criterion(out,j[-1].cuda())
            totalloss += loss*len(j[-1])
            if task == "classification":
                pred.append(torch.argmax(out, 1))
            elif task == "multilabel":
                pred.append(torch.sigmoid(out).round())
            true.append(j[-1])
            if auprc:
                #pdb.set_trace()
                sm=softmax(out)
                pts += [(sm[i][1].item(), j[-1][i].item()) for i in range(j[-1].size(0))]
        if pred:
            pred = torch.cat(pred, 0)
        true = torch.cat(true, 0)
        totals = true.shape[0]
        testloss=totalloss/totals
        if auprc:
            print("AUPRC: "+str(AUPRC(pts)))
        if task == "classification":
            print("acc: "+str(accuracy(true, pred)))
            return {'Accuracy': accuracy(true, pred)}
        elif task == "multilabel":
            print(" f1_micro: "+str(f1_score(true, pred, average="micro"))+\
                " f1_macro: "+str(f1_score(true, pred, average="macro")))
            return {'micro': f1_score(true, pred, average="micro"), 'macro': f1_score(true, pred, average="macro")}
        elif task == "regression":
            print("mse: "+str(testloss.item()))
            return {'MSE': testloss.item()}


# model: saved checkpoint filename from train
# test_dataloaders_all: test data
# dataset: the name of dataset, need to be set for testing effective robustness
# criterion: only needed for regression, put MSELoss there
# all other arguments are same as train
def test(
        model, test_dataloaders_all, dataset='default', method_name='My method', is_packed=False, criterion=nn.CrossEntropyLoss(), task="classification", auprc=False, input_to_float=True,no_robust=False):
    if no_robust:
        single_test(model,test_dataloaders_all)
        return
    def testprocess():
        single_test(model, test_dataloaders_all[list(test_dataloaders_all.keys())[0]][0], is_packed, criterion, task, auprc, input_to_float)
    all_in_one_test(testprocess, [model])
    for noisy_modality, test_dataloaders in test_dataloaders_all.items():
        print("Testing on noisy data ({})...".format(noisy_modality))
        robustness_curve = dict()
        for test_dataloader in tqdm(test_dataloaders):
            single_test_result = single_test(model, test_dataloader, is_packed, criterion, task, auprc, input_to_float)
            for k, v in single_test_result.items():
                curve = robustness_curve.get(k, [])
                curve.append(v)
                robustness_curve[k] = curve 
        for measure, robustness_result in robustness_curve.items():
            robustness_key = '{} {}'.format(dataset, noisy_modality)
            print("relative robustness ({}, {}): {}".format(noisy_modality, measure, str(relative_robustness(robustness_result, robustness_key))))
            if len(robustness_curve) != 1:
                robustness_key = '{} {}'.format(robustness_key, measure)
            print("effective robustness ({}, {}): {}".format(noisy_modality, measure, str(effective_robustness(robustness_result, robustness_key))))
            fig_name = '{}-{}-{}-{}'.format(method_name, robustness_key, noisy_modality, measure)
            single_plot(robustness_result, robustness_key, xlabel='Noise level', ylabel=measure, fig_name=fig_name, method=method_name)
            print("Plot saved as "+fig_name)
