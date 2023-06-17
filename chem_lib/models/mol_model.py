import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import json
import pandas as pd
import numpy as np

from torch_geometric.data import Data,Batch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import DataLoader
from itertools import repeat
###
from collections import OrderedDict
from .encoder import GNN_Encoder

from rdkit import Chem
from rdkit.Chem import rdmolops
import pickle
from transformers import AutoTokenizer, AutoModel

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    allowable_features = {
        'possible_atomic_num_list' : list(range(0, 119)),
        'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
        'possible_chirality_list' : [
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.rdchem.ChiralType.CHI_OTHER
        ],
        'possible_hybridization_list' : [
            Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
        ],
        'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
        'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
        'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'possible_bonds' : [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC
        ],
        'possible_bond_dirs' : [ # only for double bond stereo information
            Chem.rdchem.BondDir.NONE,
            Chem.rdchem.BondDir.ENDUPRIGHT,
            Chem.rdchem.BondDir.ENDDOWNRIGHT
        ]
    }
except:
    print('Error rdkit:')
    Chem, AllChem, allowable_features=None,None, None

class MLP(nn.Module):
    def __init__(self, inp_dim, hidden_dim,task,batch_norm=False, dropout=0.):
        super(MLP, self).__init__()
        layer_list = OrderedDict()
        in_dim = inp_dim
        layer_list['fc0'] = nn.Linear(in_dim, task)
        #torch.nn.init.zeros_(layer_list['fc0'].weight)
        self.network = nn.Sequential(layer_list)

    def forward(self, emb):
        out = self.network(emb)
        return out

class EM3P2(nn.Module):
    def __init__(self, args):
        super(EM3P2, self).__init__()
       
        self.gpu_id = args.gpu_id
        self.device = args.device
        self.mol_encoder = GNN_Encoder(num_layer=args.enc_layer, emb_dim=args.emb_dim,    JK=args.JK,
                                       drop_ratio=args.dropout, graph_pooling=args.enc_pooling, gnn_type=args.enc_gnn,
                                       batch_norm = args.enc_batch_norm)
        # get pretrained weight
        if args.pretrained:
            model_file = args.pretrained_weight_path
            if args.enc_gnn != 'gin':
                temp = model_file.split('/')
                model_file = '/'.join(temp[:-1]) +'/'+args.enc_gnn +'_'+ temp[-1]
            print('load pretrained model from', model_file)
            self.mol_encoder.from_pretrained(model_file, self.gpu_id)
        # classification layer
        self.encode_projection = MLP(inp_dim=args.emb_dim, hidden_dim=args.emb_dim , task=2, batch_norm=args.batch_norm)

    def forward(self, data, label=None):
        
        graph_emb, node_emb = self.mol_encoder(data.x, data.edge_index, data.edge_attr, data.batch)
        logits = self.encode_projection(graph_emb)
        
        return logits, node_emb, graph_emb
