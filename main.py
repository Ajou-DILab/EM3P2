'''
Title: Evidential Meta-Learning for Molecular Property Prediction
Authors:
- KP Ham, Lee Sael (sael@ajou.ac.kr) Ajou University
This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
'''
import os
print('pid:', os.getpid())
import numpy as np
import pandas as pd
import xarray as xr
import json

import random
from time import time
from parser import get_args
from chem_lib.models import EM3P2, Meta_Trainer
from chem_lib.utils import count_model_params
import collections
import torch
import gc

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def main():
    root_dir = '.'
    args = get_args(root_dir) 
    model = EM3P2(args) 
    count_model_params(model)
    model = model.to(args.device)
    trainer = Meta_Trainer(args, model)
    print("evidence : {}".format(args.evidence) )
    
    #dataframe to follow learning
    num_task = 0
    df_dict = {}
    if args.dataset == 'tox21' : #data : 7831
        num_task = 12
        for task_id in range(num_task): # 1~12
            mol_id_list = [i for i in range(7831)]
            df_dict[task_id+1]= dict.fromkeys(mol_id_list)
            
    elif args.dataset == 'sider' : #data : 1427
        num_task = 27
        for task_id in range(num_task):
            mol_id_list = [i for i in range(1427)]
            df_dict[task_id+1]= dict.fromkeys(mol_id_list)
    
    t1=time()
    print('Initial Evaluation')
    best_avg_auc=0
    
    early_stop_count = args.early_stop_count
    early_stop_temp = 0
    train_trend = {}
    for epoch in range(1, args.epochs + 1):
        print('----------------- Epoch:', epoch,' -----------------')
        torch.cuda.empty_cache()
        train_task_df,train_trend[epoch] = trainer.train_step(epoch)
        if epoch % args.eval_steps == 0 or epoch==1 or epoch ==2000 or epoch==args.epochs:
            print('Evaluation on epoch',epoch)
            test_task_df,best_avg_auc = trainer.test_step()
        if epoch % args.save_steps == 0:
            trainer.save_model()
            train_task_num = len(args.train_tasks)+1
            if args.evidence:
                
                for i in range(len(test_task_df)): 
                    for mol_id,info in test_task_df[train_task_num+i]["query"].items():
                        df_dict[train_task_num +i][mol_id] ={
                            "ID " : info["ID"],
                            "smiles":info["smiles"],
                            "labels":info["labels"],
                            "belief0":info["belief0"],
                            "belief1":info["belief1"],
                            "prob0":info["prob0"],
                            "prob1":info["prob1"],
                            "vacuity":info["vacuity"],
                            "wbv" :info["wbv"],
                            "dis" :info["dis"],
                            "cbv" :info["cbv"],
                            "pred":info["pred"],
                            "softmax":info["softmax"]
                        }
                    
                     
                json_save_path =  os.path.join(args.trial_path, f"{epoch}.json")
                with open(json_save_path,'w') as f:
                    json.dump(df_dict,f)
            else :
                for i in range(len(test_task_df)): 
                    for mol_id,info in test_task_df[train_task_num+i]["query"].items():
                        df_dict[train_task_num +i][mol_id] ={
                            "ID " : info["ID"],
                            "smiles":info["smiles"],
                            "labels":info["labels"],
                            "pred":info["pred"],
                            "softmax":info["softmax"]
                        }
                json_save_path =  os.path.join(args.trial_path, f"{epoch}.json")
                with open(json_save_path,'w') as f:
                    json.dump(df_dict,f)
                    
         
        torch.cuda.empty_cache()
        print('Time cost (min):', round((time()-t1)/60,3))
        t1=time()

    print('Train done.')
    print('Best Avg AUC:',best_avg_auc)
    
    trainer.conclude(train_trend)
    
    if args.save_logs:
        trainer.save_result_log()

if __name__ == "__main__":
    main()
