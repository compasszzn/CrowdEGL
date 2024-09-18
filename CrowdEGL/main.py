import argparse
import torch
import torch.utils.data
from data.dataset import create_dataloader
from crowd.model import Crowdegl
import os
from torch import nn, optim
import json
from tqdm import tqdm
import torch.nn.functional as F
import random
import numpy as np
import time

parser = argparse.ArgumentParser(description='Graph Mechanics Networks')

parser.add_argument('--epochs', type=int, default=5000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=5e-4, metavar='N',
                    help='learning rate')
parser.add_argument('--test_interval', type=int, default=5, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--nf', type=int, default=64, metavar='N',
                    help='hidden dim')
parser.add_argument('--n_layers', type=int, default=4, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--weight_decay', type=float, default=1e-10, metavar='N',
                    help='weight decay')
parser.add_argument('--data_dir', type=str, default='CrowdEGL/data',
                    help='Data directory.')
parser.add_argument('--dataset', type=str, default='junc',choices=['90','120','bi','uni','low','up','cor','junc'],
                    help='Data directory.')
parser.add_argument('--gpus_num', type=str, default="2",
                    help='Model name')
parser.add_argument('--lambda_rate', type=float, default=0.7, metavar='N',
                    help='rate that control equivariant')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 128)')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    device = torch.device('cuda:' + str(args.gpus_num))
else:
    device = torch.device('cpu')
loss_mse = nn.MSELoss()

print(args)


def main():
    # fix seed
    if args.seed < 0:
        seed = random.randint(0,1000)
    else:
        seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


    loader_train, loader_val, loader_test = create_dataloader(data_dir=args.data_dir,dataset=args.dataset, partition='train', batch_size=args.batch_size,
                                        shuffle=True,  num_workers=8)

    rotate_90 = torch.FloatTensor([[0, 1], [-1, 0]])
    rotate_120 = torch.FloatTensor([[-0.5, -0.866], [0.866, -0.5]])
    reflect_x = torch.FloatTensor([[-1, 0], [0, 1]])
    reflect_y = torch.FloatTensor([[1, 0], [0, -1]])
    

    if '120' in args.dataset:
        group = [torch.eye(2), rotate_120, torch.mm(rotate_120, rotate_120)]
    elif '90' in args.dataset:
        group = [torch.eye(2), rotate_90, torch.mm(rotate_90, rotate_90), torch.mm(rotate_90, torch.mm(rotate_90, rotate_90))] 
    elif 'bi' in args.dataset or 'uni' in args.dataset or 'junc' in args.dataset or 'low' in args.dataset or 'up' in args.dataset:
        group = [torch.eye(2), reflect_x] ###BIA Tjunc mouthhole   
    elif 'cor' in args.dataset:
        group = [torch.eye(2), torch.mm(rotate_90, reflect_x)]
    group = [op.to(device) for op in group]

    model = Crowdegl(input_dim=6, hidden_nf=args.nf, group=group, n_layers=args.n_layers, device=device, recurrent=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    results = {'epochs': [], 'loss': [], 'train loss': []}
    best_val_loss = 1e8
    best_test_loss = 1e8
    best_epoch = 0
    best_train_loss = 1e8

    for epoch in range(args.epochs):
        train_loss = train(model, optimizer, epoch, loader_train, args=args)
        results['train loss'].append(train_loss)
        if epoch % args.test_interval == 0:
            val_loss = train(model, optimizer, epoch, loader_val, partition='valid', backprop=False, args=args)
            test_loss = train(model, optimizer, epoch, loader_test, partition='test', backprop=False, args=args)
            results['epochs'].append(epoch)
            results['loss'].append(test_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_train_loss = train_loss
                best_epoch = epoch
            print("*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best apoch %d"
                  % (best_val_loss, best_test_loss, best_epoch))
            
            if epoch - best_epoch > 100:
                break


    return best_train_loss, best_val_loss, best_test_loss, best_epoch


def train(model, optimizer, epoch, loader, partition='train', backprop=True, args=None):
    if backprop:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch, 'loss': 0, 'coord_reg': 0, 'counter': 0, 'loss_stick': 0, 'loss_vel': 0, 'reg_loss': 0}

    for batch_idx, data in enumerate(loader):
        ped = data.ped
        loc, vel, loc_end = data.pos.to(device), data.x.to(device), data.y.to(device)
        node_type = data.node_attr.to(device)
        edges = data.edge_index.to(device)
        batch_size = loc.shape[0]

        optimizer.zero_grad()

        # helper to compute reg loss
        reg_loss = 0

        rows, cols = edges
        edge_attr = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)
        # nodes = torch.cat([loc, vel], dim=1)
        nodes = torch.cat([loc, vel, F.one_hot(node_type)], dim=1)
        loc_pred = model(nodes, edges, edge_attr)
        loc_pred = loc_pred[torch.where(node_type==0)]
        loss = loss_mse(loc_pred[:, :2], loc_end)
        res['loss'] += loss.item()*batch_size

        if backprop:
            # if epoch % 1 == 0:
            aug_loc_end = []
            for i in range(1, len(model.group)):
                g = model.group[i]
                aug_loc_end.append(torch.mm(loc_end, g))
            aug_loc_end = torch.cat(aug_loc_end, dim=1)
            reg_loss = loss_mse(loc_pred[:, 2:], aug_loc_end)
            loss += args.lambda_rate * reg_loss

            loss.backward()
            optimizer.step()
        try:
            res['reg_loss'] += reg_loss.item()*batch_size
        except:  # no reg loss (no sticks and hinges)
            pass
        res['counter'] += batch_size

    if not backprop:
        prefix = "==> "
    else:
        prefix = ""
    print('%s epoch %d avg loss: %.5f reg loss: %.5f'
          % (prefix+partition, epoch,
             res['loss'] / res['counter'], res['reg_loss'] / res['counter']))

    return res['loss'] / res['counter']


if __name__ == "__main__":
    best_train_loss, best_val_loss, best_test_loss, best_epoch = main()
    print("best_train = %.6f" % best_train_loss)
    print("best_val = %.6f" % best_val_loss)
    print("best_test = %.6f" % best_test_loss)
    print("best_epoch = %d" % best_epoch)





