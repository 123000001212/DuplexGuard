import numpy as np
import torch
import os
# from utils import *
import SleeperAgent
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

SAVE_INDICES=True
SAVE_D1=True
base_dir=os.getcwd()

unlearn_size=5000

if SAVE_INDICES:
    index_list=np.random.choice(50000,50000,replace=False)
    unlearn_indices=index_list[:unlearn_size]
    train_indices=index_list[unlearn_size:]
    print(f'Spliting Dc and Du: Size of Dc is {len(train_indices)}, Du is {len(unlearn_indices)}.')
    np.save(os.path.join(base_dir,'checkpoint/indices/unlearn_indices'),unlearn_indices)
    np.save(os.path.join(base_dir,'checkpoint/indices/train_indices'),train_indices)
else:
    unlearn_indices=np.load(os.path.join(base_dir,'checkpoint/indices/unlearn_indices.npy'))
    train_indices=np.load(os.path.join(base_dir,'checkpoint/indices/train_indices.npy'))


# craft D1 (D_a)
if SAVE_D1:
    source_class, target_class = 2,2
    intended_classes=torch.tensor([target_class]).to(device=device, dtype=torch.long)
    poison_delta,poison_indices,target_indice,retrained_model=SleeperAgent.sleeper_agent.SA(poison=None, process='D1', 
            source_class=source_class, target_class=target_class, train_indices=train_indices, budget=0.01)
    poison_indices=train_indices[poison_indices]
    torch.save(poison_delta,os.path.join(base_dir,'checkpoint/D1/poison_delta.pth'))
    torch.save(poison_indices,os.path.join(base_dir,'checkpoint/D1/poison_indices.pth'))
    torch.save(target_indice,os.path.join(base_dir,'checkpoint/D1/target_indice.pth'))
    torch.save(intended_classes,os.path.join(base_dir,'checkpoint/D1/intended_classes.pth'))
    SleeperAgent.sleeper_agent.val_D1(retrained_model,target_indice,device)
else:
    poison_delta=torch.load(os.path.join(base_dir,'checkpoint/D1/poison_delta.pth'))
    poison_indices=torch.load(os.path.join(base_dir,'checkpoint/D1/poison_indices.pth'))
    target_indice=torch.load(os.path.join(base_dir,'checkpoint/D1/target_indice.pth'))
    intended_classes=torch.load(os.path.join(base_dir,'checkpoint/D1/intended_classes.pth'))


# craft D2 (D_s)
SA_poison_delta,SA_poison_indices,SA_source_ids,retrained_model=SleeperAgent.sleeper_agent.SA((poison_delta,poison_indices),process='D2', source_class=2, target_class=2,budget=0.04)
torch.save(SA_poison_delta,os.path.join(base_dir,'checkpoint/D2/SA_poison_delta.pth'))
torch.save(SA_poison_indices,os.path.join(base_dir,'checkpoint/D2/SA_poison_indices.pth'))
torch.save(SA_source_ids,os.path.join(base_dir,'checkpoint/D2/SA_source_ids.pth'))
