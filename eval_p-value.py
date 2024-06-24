import numpy as np
import torch
import torch.utils.data as data
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# from utils import *
from SleeperAgent.forest.data.datasets import construct_datasets
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

DATASET_NAME='CIFAR10'
DATA_PATH='/home/data'
base_dir=os.getcwd()

# load indices
unlearn_indices=np.load(os.path.join(base_dir,'checkpoint/indices/unlearn_indices.npy'))
train_indices=np.load(os.path.join(base_dir,'checkpoint/indices/train_indices.npy'))

# load D1
poison_delta=torch.load(os.path.join(base_dir,'checkpoint/D1/poison_delta.pth'))
poison_indices=torch.load(os.path.join(base_dir,'checkpoint/D1/poison_indices.pth'))
target_indice=torch.load(os.path.join(base_dir,'checkpoint/D1/target_indice.pth'))
intended_classes=torch.load(os.path.join(base_dir,'checkpoint/D1/intended_classes.pth'))
trainset,testset=construct_datasets(DATASET_NAME, DATA_PATH)
targetset = data.Subset(testset, target_indice)
targetloader = torch.utils.data.DataLoader(targetset,batch_size=len(targetset))

# load D2
SA_poison_delta=torch.load(os.path.join(base_dir,'checkpoint/D2/SA_poison_delta.pth'))
SA_poison_indices=torch.load(os.path.join(base_dir,'checkpoint/D2/SA_poison_indices.pth'))
SA_poison_lookup = dict(zip(SA_poison_indices.tolist(), range(len(SA_poison_indices))))
SA_source_ids=torch.load(os.path.join(base_dir,'checkpoint/D2/SA_source_ids.pth'))

import SleeperAgent

SleeperAgent.sleeper_agent.val_pvalue((poison_delta,poison_indices),targetloader,intended_classes,SA_poison_delta,SA_poison_indices,SA_source_ids,unlearn_indices,device,
                                      D1_percentage=1.0,D2_percentage=1.0,Dunlearn_percentage=1.0)
# SleeperAgent.sleeper_agent.val_pvalue((poison_delta,poison_indices),targetloader,intended_classes,SA_poison_delta,SA_poison_indices,SA_source_ids,unlearn_indices,device,
#                                       D1_percentage=1.0,D2_percentage=0.75,Dunlearn_percentage=0.75)
# SleeperAgent.sleeper_agent.val_pvalue((poison_delta,poison_indices),targetloader,intended_classes,SA_poison_delta,SA_poison_indices,SA_source_ids,unlearn_indices,device,
#                                       D1_percentage=1.0,D2_percentage=0.5,Dunlearn_percentage=0.5)
# SleeperAgent.sleeper_agent.val_pvalue((poison_delta,poison_indices),targetloader,intended_classes,SA_poison_delta,SA_poison_indices,SA_source_ids,unlearn_indices,device,
#                                       D1_percentage=1.0,D2_percentage=0.25,Dunlearn_percentage=0.25)
# SleeperAgent.sleeper_agent.val_pvalue((poison_delta,poison_indices),targetloader,intended_classes,SA_poison_delta,SA_poison_indices,SA_source_ids,unlearn_indices,device,
#                                        D1_percentage=1.0,D2_percentage=0.0,Dunlearn_percentage=0.0)
