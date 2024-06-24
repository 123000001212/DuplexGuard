import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def change_indices(indice_to_del,SA_poison_delta, SA_poison_ids,max_indice=50000):
    indice_to_del=np.sort(indice_to_del)
    length=len(SA_poison_ids)
    same_indice=[]
    for i in range(length):
        if SA_poison_ids[i] in indice_to_del:
            same_indice.append(i)
    SA_poison_delta=np.delete(SA_poison_delta,same_indice,axis=0)
    SA_poison_ids=np.delete(SA_poison_ids,same_indice,axis=0)

    num_to_sub=np.array([0 for _ in range(max_indice)])
    for i in indice_to_del:
        num_to_sub[i:]+=1

    for i in range(len(SA_poison_ids)):
        SA_poison_ids[i]-=num_to_sub[SA_poison_ids[i]]

    return SA_poison_delta, SA_poison_ids
