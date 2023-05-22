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
    args = get_args(root_dir) ##parser로부터 필요한 파라미터들 불러와야한다.
    #
    model = EM3P2(args) # 바꿔야하는 부분(mol_model.py)
    
    count_model_params(model)

    model = model.to(args.device)
    
    trainer = Meta_Trainer(args, model)
    print("evidence : {}, random: {} ".format(args.evidence, args.random) )
    
    #dataframe to follow learning
    #nested dictionary로 시작 (hard coding)
    #json file로 저장
    num_task = 0
    df_dict = {}
    if args.dataset == 'tox21' : #data 갯수 7831
        num_task = 12
        for task_id in range(num_task): #1~12
            #df_dict[task_id+1] =  dict()#{ID: SMILES, truth_label, prob, unc,s_access_count, q_access_count}
            mol_id_list = [i for i in range(7831)]
            df_dict[task_id+1]= dict.fromkeys(mol_id_list)
            
    elif args.dataset == 'sider' : #data 갯수 1427
        num_task = 27
        for task_id in range(num_task): #1~12
            #df_dict[task_id+1] =  dict()#{ID: SMILES, truth_label, prob, unc,s_access_count, q_access_count}
            mol_id_list = [i for i in range(1427)]
            df_dict[task_id+1]= dict.fromkeys(mol_id_list)
    #print(df_dict)
    
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
        #train_trend = { task : [[vac][wbv][cbv][dis]]}
        #print("epoch",train_task_df)
        ####tox21 hard coding
        #assert len(train_task_df) == 9
        #print(train_task_df)
        if args.evidence:
            for i in range(len(train_task_df)):
                for mol_id,info in train_task_df[i]["support"].items():
                    if df_dict[i+1][mol_id] : #이미 정보가 있을 때
                        df_dict[i+1][mol_id]["sup_cnt"] = df_dict[i+1][mol_id]["sup_cnt"]+1
                    else : #None 일 때
                        df_dict[i+1][mol_id] ={
                            "smiles":info["smiles"],
                            "labels":info["labels"],
                            "probability0":info["probability0"],
                            "probability1":info["probability1"],
                            #"uncertainty":info["uncertainty"],
                            "sup_cnt" : 1,
                            "qry_cnt" : 0
                        }
                for mol_id,info in train_task_df[i]["query"].items():
                    if df_dict[i+1][mol_id] : #이미 정보가 있을 때
                        df_dict[i+1][mol_id]["qry_cnt"] = df_dict[i+1][mol_id]["qry_cnt"]+1
                    else : #None 일 때
                        df_dict[i+1][mol_id] ={
                            "smiles":info["smiles"],
                            "labels":info["labels"],
                            "probability0":info["probability0"],
                            "probability1":info["probability1"],
                            "uncertainty":info["uncertainty"],
                            "sup_cnt" : 0,
                            "qry_cnt" : 1
                        }


            #print(df_dict[task_id])
        
        ####
        if epoch % args.eval_steps == 0 or epoch==1 or epoch ==2000 or epoch==args.epochs:
            print('Evaluation on epoch',epoch)
            test_task_df,best_avg_auc = trainer.test_step()
            
#             if best_avg_auc == early_stop_temp :
#                 early_stop_count += 1
#             else :
#                 early_stop_temp = best_avg_auc
#                 early_stop_count =0

#             if early_stop_count == 9 :
#                 args.early_stop_count = 9 
#             elif early_stop_count == 10 :
#                 trainer.save_model()
#                 print("early_stopping")
#                 torch.cuda.empty_cache()
#                 print('Time cost (min):', round((time()-t1)/60,3))
#                 t1=time()
#                 break
            #print(test_task_df)
        if epoch % args.save_steps == 0:
            trainer.save_model()
            
            if args.evidence:
                for i in range(len(test_task_df)):
                    train_task_num = len(train_task_df)+1
                    for mol_id,info in test_task_df[train_task_num+i]["support"].items():
                        if df_dict[train_task_num + i][mol_id] : #이미 정보가 있을 때
                            df_dict[train_task_num +i][mol_id]["sup_cnt"] = df_dict[train_task_num +i][mol_id]["sup_cnt"]+1
                        else : #None 일 때
                            df_dict[train_task_num +i][mol_id] ={
                                "smiles":info["smiles"],
                                "labels":info["labels"],
                                "probability0":info["probability0"],
                                "probability1":info["probability1"],
                                #"uncertainty":info["uncertainty"],
                                "sup_cnt" : 1,
                                "qry_cnt" : 0
                            }


                    for mol_id,info in test_task_df[train_task_num+i]["query"].items():
                        if df_dict[train_task_num +i][mol_id] : #이미 정보가 있을 때
                            df_dict[train_task_num +i][mol_id]["qry_cnt"] = df_dict[train_task_num +i][mol_id]["qry_cnt"]+1
                        else : #None 일 때
                            df_dict[train_task_num +i][mol_id] ={
                                "smiles":info["smiles"],
                                "labels":info["labels"],
                                "probability0":info["probability0"],
                                "probability1":info["probability1"],
                                "uncertainty":info["uncertainty"],
                                "sup_cnt" : 0,
                                "qry_cnt" : 1
                            }
                json_save_path =  os.path.join(args.trial_path, f"{epoch}.json")
                with open(json_save_path,'w') as f:
                    json.dump(df_dict,f)
                    
         
        torch.cuda.empty_cache()
        print('Time cost (min):', round((time()-t1)/60,3))
        t1=time()
        #print(torch.cuda.memory_summary(device=None, abbreviated=False))

    print('Train done.')
    print('Best Avg AUC:',best_avg_auc)
    
    trainer.conclude(train_trend)
    
    if args.save_logs:
        trainer.save_result_log()

if __name__ == "__main__":
    main()
