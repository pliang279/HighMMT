import sys
import os
import numpy as np
from torch.utils.data import DataLoader
import random
import pickle
import copy
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from robustness.timeseries_robust import timeseries_robustness
from robustness.tabular_robust import tabular_robustness
from tqdm import tqdm


#task: integer between -1 and 19 inclusive, -1 means mortality task, 0-19 means icd9 task
def get_dataloader(task,batch_size=40, num_workers=0, train_shuffle=True, imputed_path='im.pk', flatten_time_series=False, tabular_robust=True, timeseries_robust=True,train_batch_size_1=False,no_robust=False,fracs=1):
  f = open(imputed_path,'rb')
  datafile = pickle.load(f)
  f.close()
  X_t = datafile['ep_tdata']
  X_s = datafile['adm_features_all']

  X_t[np.isinf(X_t)]=0
  X_t[np.isnan(X_t)]=0
  X_s[np.isinf(X_s)]=0
  X_s[np.isnan(X_s)]=0

  X_s_avg=np.average(X_s,axis=0)
  X_s_std=np.std(X_s,axis=0)
  X_t_avg=np.average(X_t,axis=(0,1))
  X_t_std=np.std(X_t,axis=(0,1))

  for i in range(len(X_s)):
    X_s[i] = (X_s[i]-X_s_avg)/X_s_std
    for j in range(len(X_t[0])):
      X_t[i][j] = (X_t[i][j]-X_t_avg)/X_t_std

  static_dim=len(X_s[0])
  timestep=len(X_t[0])
  series_dim=len(X_t[0][0])
  if flatten_time_series:
    X_t=X_t.reshape(len(X_t),timestep*series_dim)
  if task<0:
    y=datafile['adm_labels_all'][:,1]
    admlbl=datafile['adm_labels_all']
    le = len(y)
    for i in range(0,le):
      if admlbl[i][1]>0:
        y[i]=1
      elif admlbl[i][2]>0:
        y[i]=2
      elif admlbl[i][3]>0:
        y[i]=3
      elif admlbl[i][4]>0:
        y[i]=4
      elif admlbl[i][5]>0:
        y[i]=5
      else:
        y[i]=0
  else:
    y=datafile['y_icd9'][:,task]
    le = len(y)
  datasets=[(X_s[i],X_t[i],y[i]) for i in range(le)]
  random.seed(10)

  random.shuffle(datasets)

  c=[0,0]
  for i in datasets[le//10:le//5]:
      c[i[2]] += 1
  print(c)
  t_batch_size=batch_size
  if train_batch_size_1:
      t_batch_size=1

  valids = DataLoader(datasets[0:le//10], shuffle=False, num_workers=num_workers, batch_size=batch_size)
  trains = DataLoader((datasets[le//5:le//5+int((le-le//5)*fracs)]), shuffle=train_shuffle, num_workers=num_workers, batch_size=t_batch_size)
  if no_robust:
      tests = DataLoader(datasets[le//10:le//5], shuffle=False, num_workers=num_workers, batch_size=batch_size)
      return trains,valids,tests

  tests = dict()
  tests['timeseries'] = []
  for noise_level in tqdm(range(11)):
    dataset_robust = copy.deepcopy(datasets[le//10:le//5])
    if tabular_robust:
      X_s_robust = tabular_robustness([dataset_robust[i][0] for i in range(len(dataset_robust))], noise_level=noise_level/10)
    else:
      X_s_robust = [dataset_robust[i][0] for i in range(len(dataset_robust))]
    if timeseries_robust:
      X_t_robust = timeseries_robustness([[dataset_robust[i][1] for i in range(len(dataset_robust))]],noise_level=noise_level/10)[0]
    else:
      X_t_robust = [dataset_robust[i][1] for i in range(len(dataset_robust))]
    y_robust = [dataset_robust[i][2] for i in range(len(dataset_robust))]
    if flatten_time_series:
      tests['timeseries'].append(DataLoader([(X_s_robust[i],X_t_robust[i].reshape(timestep*series_dim),y_robust[i]) for i in range(len(y_robust))], shuffle=False, num_workers=num_workers, batch_size=batch_size))
    else:
      tests['timeseries'].append(DataLoader([(X_s_robust[i],X_t_robust[i],y_robust[i]) for i in range(len(y_robust))], shuffle=False, num_workers=num_workers, batch_size=batch_size))

  return trains,valids,tests
