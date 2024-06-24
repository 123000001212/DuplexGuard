"""General interface script to launch poisoning jobs."""

import torch
from torch.utils.data import Subset
import random
from PIL import Image
import torchvision
import numpy as np
from scipy import stats
import torch.nn.functional as F

import datetime
import time

from utils import change_indices

# import forest
import SleeperAgent.forest as forest

# from forest.filtering_defenses import get_defense
from SleeperAgent.forest.filtering_defenses import get_defense

from SleeperAgent.forest.data.datasets import Deltaset
torch.backends.cudnn.benchmark = forest.consts.BENCHMARK
torch.multiprocessing.set_sharing_strategy(forest.consts.SHARING_STRATEGY)

# Parse input arguments
args = forest.options_imagenet().parse_known_args()[0]

if args.deterministic:
    forest.utils.set_deterministic()

def val_pvalue(poison, target_indice,intended_classes, SA_poison_delta, SA_poison_ids,SA_source_ids,unlearn_indices,device,softmax=True,tau=0.01,save_confidence=True,
               D1_percentage=1.0,D2_percentage=1.0,Dunlearn_percentage=1.0,train_clean=False,save_path='./model'):
    print('tau=',tau)

    D1_num=int(len(poison[0])*D1_percentage)
    D2_num=int(len(SA_poison_delta)*D2_percentage)
    Dunlearn_num=int(len(unlearn_indices)*Dunlearn_percentage)
    print('Da_num =',D1_num,'; Ds_num =',D2_num,'; Du_num=',Dunlearn_num)
    poison = (poison[0][:D1_num],poison[1][:D1_num])
    SA_poison_delta=SA_poison_delta[:D2_num]
    SA_poison_ids=SA_poison_ids[:D2_num]
    
    indices_to_delete=unlearn_indices[:len(unlearn_indices)-Dunlearn_num]
    indices_to_remain=[]
    for i in range(9469):
        if i not in indices_to_delete:
            indices_to_remain.append(i)
    SA_poison_delta, SA_poison_ids=change_indices(indices_to_delete,SA_poison_delta, SA_poison_ids.numpy(),max_indice=9469)
    poison=change_indices(indices_to_delete,poison[0], poison[1],max_indice=9469)

    tau=tau*-1
    setup = forest.utils.system_startup(args)
    model = forest.Victim(args, setup=setup)
    data = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations,
                         model.defs.mixing_method, setup=setup,custom_training_data=poison)
    data.trainset.dataset.samples=[data.trainset.dataset.samples[i] for i in indices_to_remain]
    data.trainset.dataset.imgs=[data.trainset.dataset.imgs[i] for i in indices_to_remain]
    data.trainset.dataset.targets=[data.trainset.dataset.targets[i] for i in indices_to_remain]
    data.poison_ids=SA_poison_ids
    data.poison_lookup=dict(zip(SA_poison_ids.tolist(), range(len(SA_poison_ids))))
    data.source_ids=SA_source_ids
    sourceset = Subset(data.validset, indices=data.source_ids)

    '''
    # output source set
    for img,label,i in sourceset:
        dm = torch.tensor(data.trainset.data_mean)[None, :, None, None]
        ds = torch.tensor(data.trainset.data_std)[None, :, None, None]
        torchvision.utils.save_image(img*ds+dm,f'source{i}.png')
        if i not in [203,104,311]:
            break
    '''

    # # Add patch to sourceset
    # load patch
    load_patch=args.load_patch
    patch_size=args.patch_size
    dm = torch.tensor(data.trainset.data_mean)[None, :, None, None].to(device)
    ds = torch.tensor(data.trainset.data_std)[None, :, None, None].to(device)
    patch = Image.open(load_patch)
    totensor = torchvision.transforms.ToTensor()
    resize = torchvision.transforms.Resize(int(patch_size))
    patch = totensor(resize(patch))
    patch = (patch.to(device) - dm) / ds
    patch = patch.squeeze(0)
    # add patch
    print("Add patches to the source images randomely ...")
    source_delta = []
    for idx, (source_img, label, image_id) in enumerate(sourceset):
        source_img = source_img.to(device)

        patch_x = random.randrange(0,source_img.shape[1] - patch.shape[1] + 1)
        patch_y = random.randrange(0,source_img.shape[2] - patch.shape[2] + 1)
        delta_slice = torch.zeros_like(source_img).squeeze(0)
        diff_patch = patch - source_img[:, patch_x: patch_x + patch.shape[1], patch_y: patch_y + patch.shape[2]]
        delta_slice[:, patch_x: patch_x + patch.shape[1], patch_y: patch_y + patch.shape[2]] = diff_patch
        source_delta.append(delta_slice.cpu())
    data.sourceset = Deltaset(sourceset, source_delta)

    '''
    # output patched source set
    for img,label,i in data.sourceset:
        dm = torch.tensor(data.trainset.data_mean)[None, :, None, None]
        ds = torch.tensor(data.trainset.data_std)[None, :, None, None]
        torchvision.utils.save_image(img*ds+dm,f'psource{i}.png')
        if i not in [203,104,311]:
            break
    '''

    # retrain model
    model.validate(data, SA_poison_delta)

    net=model.model
    net.eval()
    # clean source: sourceset ; patched source: data.sourceset
    clean_source_loader=torch.utils.data.DataLoader(sourceset,batch_size=len(sourceset))
    patched_source_loader=torch.utils.data.DataLoader(data.sourceset,batch_size=len(data.sourceset))
    for img,label,index in clean_source_loader:
        img=img.to(device)
        source_label=label[0].item()+1 # 'label' here is source label
        clean_out=net(img)
    for img,label,index in patched_source_loader:
        img=img.to(device)
        patched_out=net(img)

    print(source_label)

    if save_confidence:
        torch.save(clean_out.detach().cpu(),save_path+f'/poison_clean_out_{D2_percentage}.pth')
        torch.save(patched_out.detach().cpu(),save_path+f'/poison_patched_out_{D2_percentage}.pth')

    if softmax:
        clean_out = F.softmax(clean_out, dim=1).detach().cpu()
        patched_out = F.softmax(patched_out, dim=1).detach().cpu()
    else:
        clean_out=clean_out.detach().cpu()
        patched_out=patched_out.detach().cpu()

    print('Testing on posion model:')
    print('Avg clean out confidence: ',clean_out.mean(dim=0))
    print('Predictions: ', torch.bincount(torch.argmax(clean_out,dim=1)))
    print('Avg patched out confidence: ',patched_out.mean(dim=0))
    print('Predictions: ', torch.bincount(torch.argmax(patched_out,dim=1)))

    source_confidence=np.array([clean_out[i][source_label].item() for i in range(len(sourceset))])
    patched_confidence=np.array([patched_out[i][source_label].item() for i in range(len(sourceset))])

    # levene = stats.levene(source_confidence, patched_confidence)          # 进行 levene 检验

    samples=source_confidence-patched_confidence
    t1, p1 = stats.ttest_ind(samples,np.ones_like(samples)*tau,alternative='less',equal_var=False)        # t检验


    # print("levene.pvalue: %f"%levene.pvalue,'\n') # >0.05 说明方差齐,可以进行t检验
    print("Poison model Pairwise T-test")
    # print("levene.pvalue: %f"%levene.pvalue)
    print("T-test: %f\n"%t1,"P-value: %f"%p1) # p-value <0.05, 两组样本均值不同，可以验证所有权；T值小于0，说明第一组数据的均值小于第二组
    #print("Poison model Wilcoxon Rank Sum test")
    #s,p=stats.ranksums(source_confidence-patched_confidence, np.ones_like(samples)*tau, alternative='less') # Wilcoxon Rank Sum test
    #print("Stats: %f\n"%s,"P-value: %f"%p)


    # -------------------------------------
    # clean model t-test
    if train_clean:
        model.validate(data, torch.zeros_like(SA_poison_delta))
        net=model.model
        net.eval()

        for img,label,index in clean_source_loader:
            img=img.to(device)
            source_label=label[0].item()-1
            clean_out=net(img)
        for img,label,index in patched_source_loader:
            img=img.to(device)
            patched_out=net(img)

        if save_confidence:
            torch.save(clean_out.detach().cpu(),save_path+f'/clean_clean_out_{D2_percentage}.pth')
            torch.save(patched_out.detach().cpu(),save_path+f'/clean_patched_out_{D2_percentage}.pth')

        if softmax:
            clean_out = F.softmax(clean_out, dim=1).detach().cpu()
            patched_out = F.softmax(patched_out, dim=1).detach().cpu()
        else:
            clean_out=clean_out.detach().cpu()
            patched_out=patched_out.detach().cpu()

        print('Testing on clean model:')
        print('Avg clean out confidence: ',clean_out.mean(dim=0))
        print('Predictions: ', torch.bincount(torch.argmax(clean_out,dim=1)))
        print('Avg patched out confidence: ',patched_out.mean(dim=0))
        print('Predictions: ', torch.bincount(torch.argmax(patched_out,dim=1)))

        source_confidence=np.array([clean_out[i][source_label].item() for i in range(len(sourceset))])
        patched_confidence=np.array([patched_out[i][source_label].item() for i in range(len(sourceset))])

        print("Clean model Pairwise T-test")
        #levene = stats.levene(source_confidence, patched_confidence)          # 进行 levene 检验
        #print("levene.pvalue: %f"%levene.pvalue)
        samples=source_confidence-patched_confidence
        t2, p2 = stats.ttest_ind(samples,np.ones_like(samples)*tau,alternative='less',equal_var=False)        # t检验
        print("T-test: %f\n"%t2,"P-value: %f"%p2) # p-value <0.05, 两组样本均值不同，可以验证所有权；T值小于0，说明第一组数据的均值小于第二组
        #print("Clean model Wilcoxon Rank Sum test")
        #s,p=stats.ranksums(source_confidence-patched_confidence, np.ones_like(samples)*tau, alternative='less') # Wilcoxon Rank Sum test
        #print("Stats: %f\n"%s,"P-value: %f"%p)


def val_D1(retrained_model,target_indice,device,softmax=True):
    # val D1
    retrained_model.eval()
    setup = forest.utils.system_startup(args)
    model = forest.Victim(args, setup=setup)
    data = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations,
                         model.defs.mixing_method, setup=setup,custom_training_data=None)
    clean_sourceset = Subset(data.validset, indices=target_indice)
    # # Add patch to sourceset
    # load patch
    load_patch=args.load_patch
    patch_size=args.patch_size
    dm = torch.tensor(data.trainset.data_mean)[None, :, None, None].to(device)
    ds = torch.tensor(data.trainset.data_std)[None, :, None, None].to(device)
    patch = Image.open(load_patch)
    totensor = torchvision.transforms.ToTensor()
    resize = torchvision.transforms.Resize(int(patch_size))
    patch = totensor(resize(patch))
    patch = (patch.to(device) - dm) / ds
    patch = patch.squeeze(0)
    # add patch
    print("Add patches to the source images randomely ...")
    source_delta = []
    for idx, (source_img, label, image_id) in enumerate(clean_sourceset):
        source_img = source_img.to(device)
        patch_x = random.randrange(0,source_img.shape[1] - patch.shape[1] + 1)
        patch_y = random.randrange(0,source_img.shape[2] - patch.shape[2] + 1)
        delta_slice = torch.zeros_like(source_img).squeeze(0)
        diff_patch = patch - source_img[:, patch_x: patch_x + patch.shape[1], patch_y: patch_y + patch.shape[2]]
        delta_slice[:, patch_x: patch_x + patch.shape[1], patch_y: patch_y + patch.shape[2]] = diff_patch
        source_delta.append(delta_slice.cpu())
    patched_sourceset = Deltaset(clean_sourceset, source_delta)
    clean_source_loader=torch.utils.data.DataLoader(clean_sourceset,batch_size=len(clean_sourceset))
    patched_source_loader=torch.utils.data.DataLoader(patched_sourceset,batch_size=len(patched_sourceset))
    for img,label,index in clean_source_loader:
        img=img.to(device)
        clean_out=retrained_model(img)
    for img,label,index in patched_source_loader:
        img=img.to(device)
        patched_out=retrained_model(img)

    if softmax:
        clean_out = F.softmax(clean_out, dim=1).detach().cpu()
        patched_out = F.softmax(patched_out, dim=1).detach().cpu()
    else:
        clean_out=clean_out.detach().cpu()
        patched_out=patched_out.detach().cpu()

    print('Testing on retrained_model:')
    print('Avg clean out confidence: ',clean_out.mean(dim=0))
    print('Predictions: ', torch.bincount(torch.argmax(clean_out,dim=1)))
    print('Avg patched out confidence: ',patched_out.mean(dim=0))
    print('Predictions: ', torch.bincount(torch.argmax(patched_out,dim=1)))



def val_D1_retrain(poison,source_ids,device):
    setup = forest.utils.system_startup(args)
    model = forest.Victim(args, setup=setup)
    data = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations,
                         model.defs.mixing_method, setup=setup,custom_training_data=poison)
    sourceset = Subset(data.validset, indices=source_ids)
    # # Add patch to sourceset
    # load patch
    load_patch=args.load_patch
    patch_size=args.patch_size
    dm = torch.tensor(data.trainset.data_mean)[None, :, None, None].to(device)
    ds = torch.tensor(data.trainset.data_std)[None, :, None, None].to(device)
    patch = Image.open(load_patch)
    totensor = torchvision.transforms.ToTensor()
    resize = torchvision.transforms.Resize(int(patch_size))
    patch = totensor(resize(patch))
    patch = (patch.to(device) - dm) / ds
    patch = patch.squeeze(0)
    # add patch
    print("Add patches to the source images randomely ...")
    source_delta = []
    for idx, (source_img, label, image_id) in enumerate(sourceset):
        source_img = source_img.to(device)

        patch_x = random.randrange(0,source_img.shape[1] - patch.shape[1] + 1)
        patch_y = random.randrange(0,source_img.shape[2] - patch.shape[2] + 1)
        delta_slice = torch.zeros_like(source_img).squeeze(0)
        diff_patch = patch - source_img[:, patch_x: patch_x + patch.shape[1], patch_y: patch_y + patch.shape[2]]
        delta_slice[:, patch_x: patch_x + patch.shape[1], patch_y: patch_y + patch.shape[2]] = diff_patch
        source_delta.append(delta_slice.cpu())
    data.sourceset = Deltaset(sourceset, source_delta)
    # val D2
    model.validate(data, None)

    clean_source_loader=torch.utils.data.DataLoader(sourceset,batch_size=len(sourceset))
    patched_source_loader=torch.utils.data.DataLoader(data.sourceset,batch_size=len(data.sourceset))
    net=model.model
    net.eval()
    for img,label,index in clean_source_loader:
        img=img.to(device)
        clean_out=net(img)
    for img,label,index in patched_source_loader:
        img=img.to(device)
        patched_out=net(img)

    clean_out = F.softmax(clean_out, dim=1).detach().cpu()
    patched_out = F.softmax(patched_out, dim=1).detach().cpu()

    print('Testing on posion model:')
    print('Avg clean out confidence: ',clean_out.mean(dim=0))
    print('Predictions: ', torch.bincount(torch.argmax(clean_out,dim=1)))
    print('Avg patched out confidence: ',patched_out.mean(dim=0))
    print('Predictions: ', torch.bincount(torch.argmax(patched_out,dim=1)))

def SA(poison, process=None, source_class=None, target_class=None, train_indices=None, budget=0.01):
    print(f"-------------{'Da' if process=='D1' else 'Ds'} generation started.-------------------------")
    args.budget = budget
    setup = forest.utils.system_startup(args)
    model = forest.Victim(args, setup=setup)
    data = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations,
                         model.defs.mixing_method, setup=setup,
                         custom_training_data=poison, train_indices=train_indices, source_class=source_class, target_class=target_class)
    witch = forest.Witch(args, setup=setup)
    if args.backdoor_poisoning:
        witch.patch_sources(data)
    start_time = time.time()
    stats_clean = model.train(data, max_epoch=args.max_epoch)
    train_time = time.time()
    if args.poison_selection_strategy != None:
        data.select_poisons(model, args.poison_selection_strategy)
    poison_delta = witch.brew(model, data,process=process)
    craft_time = time.time()
    # retrain from-scratch
    model.validate(data, poison_delta)
    test_time = time.time()
    # Export
    if args.save is not None:
        data.export_poison(poison_delta, path=args.poison_path, mode=args.save)

    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print('---------------------------------------------------')
    print(f'Finished computations with train time: {str(datetime.timedelta(seconds=train_time - start_time))}')
    print(f'--------------------------- craft time: {str(datetime.timedelta(seconds=craft_time - train_time))}')
    print(f'--------------------------- test time: {str(datetime.timedelta(seconds=test_time - craft_time))}')
    print(f'-------------{process} generation finished.-------------------------')

    return poison_delta, data.poison_ids, data.source_ids, model.model

if __name__ == "__main__":

    setup = forest.utils.system_startup(args)

    model = forest.Victim(args, setup=setup)
    data = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations,
                         model.defs.mixing_method, setup=setup)
    witch = forest.Witch(args, setup=setup)

    if args.backdoor_poisoning:
        witch.patch_sources(data)

    start_time = time.time()
    if args.pretrained_model:
        print('Loading pretrained model...')
        stats_clean = None
    elif args.skip_clean_training:
        print('Skipping clean training...')
        stats_clean = None
    else:
        stats_clean = model.train(data, max_epoch=args.max_epoch)
    train_time = time.time()

    if args.poison_selection_strategy != None:
        data.select_poisons(model, args.poison_selection_strategy)

    poison_delta = witch.brew(model, data)
    craft_time = time.time()

    # Optional: apply a filtering defense
    if args.filter_defense != '':
        if args.scenario == 'from-scratch':
            model.validate(data, poison_delta)
        print('Attempting to filter poison images...')
        defense = get_defense(args)
        clean_ids = defense(data, model, poison_delta)
        poison_ids = set(range(len(data.trainset))) - set(clean_ids)
        removed_images = len(data.trainset) - len(clean_ids)
        removed_poisons = len(set(data.poison_ids.tolist()) & poison_ids)

        data.reset_trainset(clean_ids)
        print(f'Filtered {removed_images} images out of {len(data.trainset.dataset)}. {removed_poisons} were poisons.')
        filter_stats = dict(removed_poisons=removed_poisons, removed_images_total=removed_images)
    else:
        filter_stats = dict()

    if not args.pretrained_model and args.retrain_from_init:
        stats_rerun = model.retrain(data, poison_delta)
    else:
        stats_rerun = None

    if args.vnet is not None:  # Validate the transfer model given by args.vnet
        train_net = args.net
        args.net = args.vnet
        args.ensemble = len(args.vnet)
        if args.vruns > 0:
            model = forest.Victim(args, setup=setup)  # this instantiates a new model with a different architecture
            stats_results = model.validate(data, poison_delta, val_max_epoch=args.val_max_epoch)
        else:
            stats_results = None
        args.net = train_net
    else:  # Validate the main model
        if args.vruns > 0:
            stats_results = model.validate(data, poison_delta, val_max_epoch=args.val_max_epoch)
        else:
            stats_results = None
    test_time = time.time()

    timestamps = dict(train_time=str(datetime.timedelta(seconds=train_time - start_time)).replace(',', ''),
                      craft_time=str(datetime.timedelta(seconds=craft_time - train_time)).replace(',', ''),
                      test_time=str(datetime.timedelta(seconds=test_time - craft_time)).replace(',', ''))
    # Save run to table
    results = (stats_clean, stats_rerun, stats_results)
    forest.utils.record_results(data, witch.stat_optimal_loss, results,
                                args, model.defs, model.model_init_seed, extra_stats={**filter_stats, **timestamps})

    # Export
    if args.save is not None:
        data.export_poison(poison_delta, path=args.poison_path, mode=args.save)

    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print('---------------------------------------------------')
    print(f'Finished computations with train time: {str(datetime.timedelta(seconds=train_time - start_time))}')
    print(f'--------------------------- craft time: {str(datetime.timedelta(seconds=craft_time - train_time))}')
    print(f'--------------------------- test time: {str(datetime.timedelta(seconds=test_time - craft_time))}')
    print('-------------Job finished.-------------------------')
