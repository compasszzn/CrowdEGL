import torch
from torch import nn
import torch.nn.functional as F
from models.gcl import GCL
import numpy as np

class Crowdegl(nn.Module):
    def __init__(self, input_dim, hidden_nf, group, device='cpu', act_fn=nn.SiLU(), n_layers=4, attention=0, recurrent=False, graph_decoder=True):
        super(Crowdegl, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.graph_decoder = graph_decoder
        
        self.add_module("dgcl_%d" % 0, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_nf=1,
                                              act_fn=act_fn, attention=attention, recurrent=recurrent))
        
        self.add_module("dgcl_%d" % 1, GCL(self.hidden_nf, 2 * len(group), self.hidden_nf, edges_in_nf=1,
                                              act_fn=act_fn, attention=attention, recurrent=False))
        
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_nf=1,
                                              act_fn=act_fn, attention=attention, recurrent=recurrent))

        self.decoder = nn.Sequential(nn.Linear(hidden_nf, hidden_nf),
                              act_fn,
                              nn.Linear(hidden_nf, 2 * len(group)))
        self.embedding = nn.Sequential(nn.Linear(input_dim, hidden_nf))
        self.group = group

        self.group_mlp = nn.Sequential(
            nn.Linear(hidden_nf * len(self.group), hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf))

        self.to(self.device)

    def forward(self, nodes, edges, edge_attr=None):
        h = []
        for g in self.group:
            loc, vel, node_type = nodes[:, :2], nodes[:, 2:4], nodes[:, 4:] 
            loc, vel = torch.mm(loc, g), torch.mm(vel, g)
            h.append(self.embedding(torch.cat([loc, vel, node_type], dim=1)))
        h = torch.cat(h, dim=1)
        h = self.group_mlp(h)
            
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr)
        x = h
#         pred = []
#         for g in self.group:
#             loc, vel, node_type = nodes[:, :2], nodes[:, 2:4], nodes[:, 4:] 
#             loc, vel = torch.mm(loc, g), torch.mm(vel, g)
#             h = self.embedding(torch.cat([loc, vel, node_type], dim=1))
#             for i in range(0, self.n_layers):
#                 h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr)
#             pred.append(h)
            
#         pred = torch.cat(pred, dim=1)
#         x = self.group_mlp(pred)
        
        if self.graph_decoder:
            x, _ = self._modules["dgcl_%d" % 0](x, edges, edge_attr=edge_attr)
            x, _ = self._modules["dgcl_%d" % 1](x, edges, edge_attr=edge_attr)
        else:
            x = self.decoder(x)

        return x


