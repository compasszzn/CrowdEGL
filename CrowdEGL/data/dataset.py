import numpy as np
import torch
import pickle as pkl
import os
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import radius_graph

def create_dataloader(data_dir,dataset, partition, batch_size=32, shuffle=True, num_workers=8):
    train_par, val_par, test_par = 0.7, 0.1, 0.2
    Data_list = []

    if dataset=='90':
        traj_dir = os.path.join(data_dir, 'crossing90')
        ob_dir = os.path.join(data_dir, 'crossing90_obstacle')
    elif dataset=='120':
        traj_dir = os.path.join(data_dir, 'crossing120_10')
        ob_dir = os.path.join(data_dir, 'crossing120_obstacle')         

    files = os.listdir(ob_dir)
    ob_pos = np.load(os.path.join(ob_dir, files[0]), allow_pickle=True)
    ob_pos = np.transpose(np.array(ob_pos))
    ob_pos = torch.Tensor(ob_pos)
    if dataset=='low' or dataset=='up' or dataset=='junc' or dataset=='cor':
        ob_pos=ob_pos/100

    # dir = os.path.join(data_dir, partition)
    files = os.listdir(traj_dir)
    print(files)
    for file in files:
        samples = np.load(os.path.join(traj_dir, file), allow_pickle=True)
        file_name = file.split('.')[0]
        samples = samples.item()[file_name]
        for frames in samples:
            start_frame, end_frame = frames.keys()
            if start_frame > end_frame:
                start_frame, end_frame = end_frame, start_frame
            start_pos, end_pos = frames[start_frame][:, 1:3], frames[end_frame][:, 1:3]
            start_vel, end_vel = frames[start_frame][:, 3:5], frames[end_frame][:, 3:5]
            acc = frames[end_frame][:, -2:] * 320 
            if dataset=='90' or dataset=='120':
                start_pos[:, 1] = start_pos[:, 1] - 1    ## crossing
                end_pos[:, 1] = end_pos[:, 1] - 1   ###
                ob_pos[:, 1]=ob_pos[:, 1] - 1
            else:
                raise Exception("Wrong dataset")


            node_feat = [0] * start_pos.shape[0] + [1] * ob_pos.shape[0]
            node_feat = torch.Tensor(node_feat).long()
            ped = start_pos.shape[0]

            start_pos = torch.cat([start_pos, ob_pos], dim=0)

            ob_vel = [0, 0] * ob_pos.shape[0]
            ob_vel = torch.tensor(ob_vel).reshape(ob_pos.shape[0], 2)
            start_vel = torch.cat([start_vel, ob_vel], dim=0)
            edges = radius_graph(start_pos, r=1, max_num_neighbors=100, loop=False)
            # edges, edge_attr = create_graph(start_pos)
            graph = Data(x=start_vel, edge_index=edges, pos=start_pos, acc=acc, node_attr=node_feat, ped=ped, y=end_pos)
            Data_list.append(graph)

    dataset_size = len(Data_list)

    np.random.seed(100)
    train_idx = np.random.choice(np.arange(dataset_size), size=int(train_par * dataset_size), replace=False)
    flag = np.zeros(dataset_size)
    for _ in train_idx:
        flag[_] = 1
    rest = [_ for _ in range(dataset_size) if not flag[_]]
    val_idx = np.random.choice(rest, size=int(val_par * dataset_size), replace=False)
    for _ in val_idx:
        flag[_] = 1
    rest = [_ for _ in range(dataset_size) if not flag[_]]
    test_idx = np.random.choice(rest, size=int(test_par * dataset_size), replace=False)

    print(len(train_idx), len(test_idx), len(val_idx))
    # ddd
    
    train_set = [Data_list[i] for i in train_idx]
    val_set = [Data_list[i] for i in val_idx]
    test_set = [Data_list[i] for i in test_idx]


    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return trainloader, validloader, testloader

