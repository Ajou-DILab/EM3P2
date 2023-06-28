'''
Title: Evidential Meta-Learning for Molecular Property Prediction
Authors:
- KP Ham, Lee Sael (sael@ajou.ac.kr) Ajou University

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
'''
import random
import os
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional import auroc
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
import collections
from learn2learn.utils import detach_module
from .maml import MAML
from ..datasets import sample_meta_datasets, sample_test_datasets, MoleculeDataset,sample_all
from ..utils import Logger
from chem_lib.loss_func import Evidence_Classifier, AvULoss
class attention(nn.Module):
    def __init__(self, dim):
        super(attention, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        # x = x.view(x.size(0), -1)
        x = self.layers(x)
        x = self.softmax(torch.transpose(x, 1, 0))
        return x
    

class Meta_Trainer(nn.Module):
    def __init__(self, args, model):
        super(Meta_Trainer, self).__init__()
        self.args = args
        self.model = MAML(model, lr=args.inner_lr, first_order=not args.second_order, anil=False, allow_unused=True)
        #optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.meta_lr, weight_decay=args.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.args.epochs, eta_min=self.args.min_learning_rate)
        #dataload
        self.dataset = args.dataset
        self.test_dataset = args.test_dataset if args.test_dataset is not None else args.dataset
        self.data_dir = args.data_dir
        self.train_tasks = args.train_tasks
        self.test_tasks = args.test_tasks
        self.n_shot_train = args.n_shot_train
        self.n_shot_test = args.n_shot_test
        self.n_query = args.n_query
        self.n_way = 2
        #evidence
        self.evidence = args.evidence
        self.temperature = args.temperature
        self.device = args.device
        self.emb_dim = args.emb_dim
        self.batch_task = args.batch_task
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.inner_update_step = args.inner_update_step
        self.attention = attention(self.emb_dim).to(self.device)
        self.current_epoch = 0
        self.evidence_classifier = Evidence_Classifier(self.args).to(self.device)
        self.avucloss = AvULoss().to(self.device)
        self.freeze = args.freeze
        self.trial_path = args.trial_path
        #criterion은 두가지 BCEloss / Dirichlet_loss
        if self.evidence :
            self.criterion1 = nn.CrossEntropyLoss().to(args.device)
            self.criterion2 = self.evidence_classifier.calc_loss_vac_bel
            
        else :
            self.criterion = nn.CrossEntropyLoss().to(args.device)
        

        trial_name = self.dataset + '_' + self.test_dataset + '@' + args.enc_gnn
        print(trial_name)
        logger = Logger(self.trial_path + '/results.txt', title=trial_name)
        log_names = ['Epoch']
        log_names += ['AUC-' + str(t) for t in args.test_tasks]
        log_names += ['AUC-Avg', 'AUC-Mid','AUC-Best']

        logger.set_names(log_names)
        self.logger = logger
         
        preload_train_data = {}
        if args.preload_train_data:
            print('preload train data')
            for task in self.train_tasks:
                dataset = MoleculeDataset(self.data_dir + self.dataset + "/new/" + str(task + 1),dataset=self.dataset)
                preload_train_data[task] = dataset
        preload_test_data = {}
        if args.preload_test_data:
            print('preload_test_data')
            for task in self.test_tasks:
                dataset = MoleculeDataset(self.data_dir + self.test_dataset + "/new/" + str(task + 1),
                                          dataset=self.test_dataset)
                preload_test_data[task] = dataset
        self.preload_train_data = preload_train_data
        self.preload_test_data = preload_test_data
        if 'train' in self.dataset and args.support_valid:
            val_data_name = self.dataset.replace('train','valid')
            print('preload_valid_data')
            preload_val_data = {}
            for task in self.train_tasks:
                dataset = MoleculeDataset(self.data_dir + val_data_name + "/new/" + str(task + 1),
                                          dataset=val_data_name)
                preload_val_data[task] = dataset
            self.preload_valid_data = preload_val_data

        self.train_epoch = 0
        self.best_auc=0 
        self.res_logs=[]

    def loader_to_samples(self, data):
        loader = DataLoader(data, batch_size=len(data), shuffle=False, num_workers=0)
        
        for samples in loader:
            samples=samples.to(self.device)
            return samples

    
    def get_data_sample(self, task_id, train=True, flag=True):
        
        if train:
            task = self.train_tasks[task_id]
            
            if task in self.preload_train_data:
                dataset = self.preload_train_data[task]
            else:
                dataset = MoleculeDataset(self.data_dir + self.dataset + "/new/" + str(task + 1), dataset=self.dataset)
            
            s_data, q_data = sample_meta_datasets(dataset, self.dataset, task,self.n_shot_train, self.n_query,self.args.random)
            
            s_data = self.loader_to_samples(s_data)
            q_data = self.loader_to_samples(q_data)
            
            adapt_data = {'s_data': s_data, 's_label': s_data.y, 'q_data': q_data, 'q_label': q_data.y}
            eval_data = { }
            
        else:
            
            task = self.test_tasks[task_id]
            if task in self.preload_test_data:
                dataset = self.preload_test_data[task]
            else:
                dataset = MoleculeDataset(self.data_dir + self.test_dataset + "/new/" + str(task + 1),
                                          dataset=self.test_dataset)
            
            s_data, q_data = sample_test_datasets(dataset, self.test_dataset, task, self.n_shot_test, self.n_query, self.update_step_test,flag=flag)
            
            s_data = self.loader_to_samples(s_data)
            
            q_data = self.loader_to_samples(q_data)
            
            if not flag:
                print(s_data.id)
                print(q_data.id)
            
            if flag :
                q_loader = DataLoader(q_data, batch_size=self.n_query, shuffle=True, num_workers=0)
            
                q_loader_adapt = DataLoader(s_data, batch_size=len(s_data), shuffle=True, num_workers=0)
            else :
                q_loader = DataLoader(q_data, batch_size=self.n_query, shuffle=False, num_workers=0)
            
                q_loader_adapt = DataLoader(q_data_adapt, batch_size=self.n_query, shuffle=False, num_workers=0)
                
            adapt_data = {'s_data': s_data, 's_label': s_data.y, 'q_data': q_data, 'q_label':q_data.y}
            eval_data = {'s_data': s_data, 's_label': s_data.y, 'q_data': q_data, 'q_label':q_data.y}
                
           
        return adapt_data, eval_data

        
    def get_adaptable_weights(self, model, adapt_weight=None):
        if adapt_weight is None:
            adapt_weight = self.args.adapt_weight
        fenc = lambda x: x[0]== 'mol_encoder'
        menc = lambda x: x[0]== 'encode_projection'

        if adapt_weight==0:
            flag=lambda x: not fenc(x)

        else:
            flag= lambda x: True
            
        if self.train_epoch < self.args.meta_warm_step or self.train_epoch>self.args.meta_warm_step2:
            adaptable_weights = None
        else:
            adaptable_weights = []
            adaptable_names=[]
            for name, p in model.module.named_parameters():
                names=name.split('.')
                if self.freeze :
                    if name == 'mol_encoder':
                        pass
                    else :
                        adaptable_weights.append(p)
                        adaptable_names.append(name)
                    
                else:
                    adaptable_weights.append(p)
                    adaptable_names.append(name)
            
        return adaptable_weights

    def get_loss(self, model, data, label,task_id ,train=True, query=False,test=False):
        num_classes = len(self.train_tasks+self.test_tasks)
        one_hot = torch.zeros(num_classes)
        one_hot[task_id] = 1
        
        if train :
            if self.evidence :
                #pred: logit
                if query : # outer-loop train
                    task_info = {}
                    
                    pred, node_emb, graph_emb= model(data) 
                    
                    evidence = F.softplus(pred/self.temperature)
                    
                    alpha = evidence + 1

                    prob = alpha / torch.sum(alpha.detach(),1,keepdim=True)
                    
                    uncertainty = self.n_way/torch.sum(alpha.detach(),1)

                    dirichlet_strength = torch.sum(alpha, dim=1)
                    
                    dirichlet_strength = dirichlet_strength.reshape((-1, 1))
                    # Belief
                    belief = evidence / dirichlet_strength

                    one_task_emb = torch.div(torch.mean(node_emb,0), 2.0)
                    #one_task_emb = torch.mean(node_emb,0).unsqueeze(0)
                    #one_task_emb = torch.mean(task_graph,0)

                    loss, vacuity, wbv, cbv, dis = self.criterion2(pred/self.temperature, label,query_set=True)
                    
                    
                    return loss,prob, vacuity,wbv,cbv,dis, one_task_emb,belief
                
                
                else : #INNER_LOOP (ADAPTION)
                    pred, node_emb, graph_emb = model(data)
                    prob = F.softmax(pred)
                    loss = self.criterion1(prob,label)
                    return loss

            else :
                pred, node_emb,graph_emb = model(data)
                pred = F.softmax(pred)
                loss = self.criterion(pred, label)
                one_task_emb = torch.div(torch.mean(node_emb,0), 2.0)
                return loss, one_task_emb
            
            
        else : #META-TEST -> Evaluation
            
            
            if self.evidence :
                pred, node_emb,graph_emb= model(data)
                evidence = F.softplus(pred/self.temperature)
                alpha = evidence + 1
                prob = alpha / torch.sum(alpha.detach(),1,keepdim=True)
                uncertainty = self.n_way/torch.sum(alpha.detach(),1)#vacuity
                _, vacuity, wbv, cbv, dis = self.criterion2(pred/self.temperature,label)
                dirichlet_strength = torch.sum(alpha, dim=1)
                dirichlet_strength = dirichlet_strength.reshape((-1, 1))
                belief = evidence / dirichlet_strength
                
                
                return pred, prob, vacuity, wbv, cbv, dis,belief
            
            else :
                
                pred, node_emb,graph_emb,_,_ = model(data)
                one_task_emb = torch.div(torch.mean(node_emb,0), 2.0)
                return pred, one_task_emb
            

    def train_step(self,epoch):
        torch.cuda.empty_cache()
        self.train_epoch += 1
        self.current_epoch = epoch
        self.evidence_classifier.current_epoch = self.current_epoch
        task_id_list = list(range(len(self.train_tasks)))
        
        if self.batch_task > 0:
            batch_task = min(self.batch_task, len(task_id_list))
            task_id_list = random.sample(task_id_list, batch_task)
        
        data_batches={}
        train_task_df = {} # 각각의 task에 대한 df suppor, query가 key
        for task_id in task_id_list:
            db = self.get_data_sample(task_id, train=True)
            data_batches[task_id]=db
            train_task_df[task_id] = dict()
        ### data loaded
        
        for k in range(self.update_step):
            losses_eval = []
            task_emb = []
            task_unc = []
            
            #get_trend에 필요한 요소 vacuity, wbv, cov, dis
            vac_epoch = []
            wbv_epoch = []
            cov_epoch = []
            dis_epoch = []
            task_epoch = {}
            
            
            for task_id in task_id_list:
                print(task_id)
                
                vac_task = []
                wbv_task = []
                cov_task = []
                dis_task = []
                
                
                train_data, _ = data_batches[task_id]
                model = self.model.clone()
                model.train()
                
                adaptable_weights = self.get_adaptable_weights(model)
                #inner-loop adaption
                for inner_step in range(self.inner_update_step): #1번(adopt per task)
                    adaptable_weights = self.get_adaptable_weights(model)
                    if self.evidence:
                        loss_adapt= self.get_loss(model, train_data['s_data'],train_data['s_label'],task_id,train=True,query=False)
                    else :
                        loss_adapt,_ = self.get_loss(model, train_data['s_data'],train_data['s_label'],task_id,train=True,query=False)
                    model.adapt(loss_adapt, adaptable_weights = adaptable_weights)
                    
                    #outer loop query loss acquisition
                    if self.evidence:
                        
                        loss_eval,qprob,quncertainty,wbv,cbv,dis, one_task_emb,_ = self.get_loss(model, train_data['q_data'], train_data['q_label'],task_id,train=True,query=True)
                        
                        #For task uncertainty
                        av_q_vac = torch.mean(quncertainty)
                        avg_wbv = torch.mean(wbv)
                        av_q_dis = torch.mean(dis)
                        
                        # For train trend acquisition
                        vac_item = [np.round(i.item(),4) for i in quncertainty]
                        wbv_item = [np.round(i.item(),4) for i in wbv]
                        cbv_item = [np.round(i.item(),4) for i in cbv]
                        dis_item = [np.round(i.item(),4) for i in dis]
                        task_epoch[task_id] = [vac_item,wbv_item,cbv_item,dis_item]
                        
                        
                    else :
                        loss_eval,one_task_emb = self.get_loss(model, train_data['q_data'], train_data['q_label'],task_id,train=True,query=True)
                    task_emb.append(one_task_emb)
                    losses_eval.append(loss_eval)

            
            losses_eval = torch.stack(losses_eval)
            task_emb = torch.stack(task_emb)
            task_emb = task_emb.detach()
            task_weight = self.attention(task_emb)
            losses_eval = torch.sum(task_weight*losses_eval)
            losses_eval = losses_eval / len(task_id_list)
            self.optimizer.zero_grad()
            losses_eval.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            
            print('Train Epoch:',self.train_epoch,', train update step:', k, ', loss_eval:', losses_eval.item())
            
            
        task_epoch = collections.OrderedDict(sorted(task_epoch.items()))    
        return train_task_df,task_epoch

    def test_step(self,flag=True):
        step_results={'query_preds':[], 'query_labels':[],'task_index':[]}
        auc_scores = []
        unc = []
        test_task_df = dict()
        for task_id in range(len(self.test_tasks)):
            adapt_data, eval_data = self.get_data_sample(task_id, train=False, flag=flag)
            model = self.model.clone()
            test_task_num = task_id+len(self.train_tasks)+1
            print("test_task : ", test_task_num)
            test_task_df[test_task_num] = dict()
            
            if self.update_step_test>0:
                model.train()
                #Meta-Test inner_loop Adaption
                for i in range(self.update_step_test):
                    print("update_test: ", i)
                    cur_adapt_data = {'s_data': adapt_data['s_data'], 's_label': adapt_data['s_label']}
                    adaptable_weights = self.get_adaptable_weights(model)
                    if self.evidence:
                        loss_adapt,sprob,_,_,_,_,_,_= self.get_loss(model, cur_adapt_data['s_data'], cur_adapt_data['s_label'],task_id,train=True,query=True)
                    else:
                        loss_adapt,sprob = self.get_loss(model, cur_adapt_data['s_data'], cur_adapt_data['s_label'],task_id,train=True,query=False)
                    
                    model.adapt(loss_adapt, adaptable_weights=adaptable_weights)
                    if i>= self.update_step_test-1:
                        break
            #Meta-Test evaluation
            model.eval()
            with torch.no_grad():
                if self.evidence: 
                    q_pred, qprob, quncertainty,qwbv,qcbv,qdis,qbelief = self.get_loss(model,eval_data['q_data'],eval_data['q_label'],task_id, train=False)
                    s_pred,sprob, suncertainty,swbv,scbv,sdis, sbelief = self.get_loss(model, cur_adapt_data['s_data'], cur_adapt_data['s_label'],task_id,train=False)
                    y_score = qprob[:,1]
                    _,qy_pred = torch.max(qprob,1)
                    _,sy_pred = torch.max(sprob,1)
                    #test_task data frame
                    smqy = F.softmax(q_pred,dim=-1)[:,1]
                    smsy = F.softmax(s_pred,dim=-1)[:,1]
                    test_task_df[test_task_num]["support"] = {}
                    test_task_df[test_task_num]["query"] = {}
                    print('recording...')
                    #recording support
                    for i in range(self.n_shot_train*2):
                        test_task_df[test_task_num]["support"][cur_adapt_data['s_data'].id[i].item()]={
                            "ID" : cur_adapt_data['s_data'].id[i].item(),
                            "smiles":cur_adapt_data['s_data'].smiles[i],
                            "labels":cur_adapt_data['s_data'].y[i].item(),
                            "belief0":np.round(sbelief[i][0].item(),4),
                            "belief1":np.round(sbelief[i][1].item(),4),
                            "prob0":np.round(sprob[i][0].item(),4),
                            "prob1": np.round(sprob[i][1].item(),4),
                            "vacuity":np.round(suncertainty[i].item(),4),
                            "wbv" : np.round(swbv[i].item(),4),
                            "dis" : np.round(sdis[i].item(),4),
                            "cbv" : np.round(scbv[i].item(),4),
                            "pred": sy_pred[i].item(),
                            "softmax" : np.round(smsy[i].item(),4)
                        }
                    #recording query
                    for i in range(self.n_shot_train*2):
                        test_task_df[test_task_num]["query"][cur_adapt_data['s_data'].id[i].item()]={
                            "ID" : cur_adapt_data['s_data'].id[i].item(),
                            "smiles":cur_adapt_data['s_data'].smiles[i],
                            "labels":cur_adapt_data['s_data'].y[i].item(),
                            "belief0":np.round(sbelief[i][0].item(),4),
                            "belief1":np.round(sbelief[i][1].item(),4),
                            "prob0":np.round(sprob[i][0].item(),4),
                            "prob1": np.round(sprob[i][1].item(),4),
                            "vacuity":np.round(suncertainty[i].item(),4),
                            "wbv" : np.round(swbv[i].item(),4),
                            "dis" : np.round(sdis[i].item(),4),
                            "cbv" : np.round(scbv[i].item(),4),
                            "pred": sy_pred[i].item(),
                            "softmax" : np.round(smsy[i].item(),4)
                        }
                        
                    for i in range(len(eval_data['q_data'].id)):
                        test_task_df[test_task_num]["query"][eval_data['q_data'].id[i].item()]={
                            "ID" : eval_data['q_data'].id[i].item(),
                            "smiles":eval_data['q_data'].smiles[i],
                            "labels":eval_data['q_data'].y[i].item(),
                            "belief0":np.round(qbelief[i][0].item(),4),
                            "belief1":np.round(qbelief[i][1].item(),4),
                            "prob0":np.round(qprob[i][0].item(),4),
                            "prob1": np.round(qprob[i][1].item(),4),
                            "vacuity":np.round(quncertainty[i].item(),4),
                            "wbv" : np.round(qwbv[i].item(),4),
                            "dis" : np.round(qdis[i].item(),4),
                            "cbv" : np.round(qcbv[i].item(),4),
                            "pred": qy_pred[i].item(),
                            "softmax" : np.round(smqy[i].item(),4)
                        }
                        
                        
                else:# normal softmax
                    q_pred, _ = self.get_loss(model,eval_data['q_data'],eval_data['q_label'],task_id, train=False)
                    s_pred, _ = self.get_loss(model,eval_data['s_data'],eval_data['s_label'],task_id, train=False)
                    y_score = F.softmax(q_pred,dim=-1)
                    qprob = y_score
                    sprob = F.softmax(s_pred,dim=-1)
                    #_,y_pred = torch.max(y_score,1)
                    y_score = y_score.detach()[:,1]
                    _,qy_pred = torch.max(qprob,1)
                    _,sy_pred = torch.max(sprob,1)
                    smqy = F.softmax(q_pred,dim=-1)[:,1]
                    smsy = F.softmax(s_pred,dim=-1)[:,1]
                    test_task_df[test_task_num]["support"] = {}
                    test_task_df[test_task_num]["query"] = {}
                    print('recording...')
                    for i in range(self.n_shot_train*2):
                        test_task_df[test_task_num]["support"][cur_adapt_data['s_data'].id[i].item()]={
                            "ID" : eval_data['s_data'].id[i].item(),
                            "smiles":cur_adapt_data['s_data'].smiles[i],
                            "labels":cur_adapt_data['s_data'].y[i].item(),
                            "pred": sy_pred[i].item(),
                            "softmax" : np.round(smsy[i].item(),4)
                        }
                    for i in range(self.n_shot_train*2):
                        test_task_df[test_task_num]["query"][cur_adapt_data['s_data'].id[i].item()]={
                            "ID" : eval_data['s_data'].id[i].item(),
                            "smiles":cur_adapt_data['s_data'].smiles[i],
                            "labels":cur_adapt_data['s_data'].y[i].item(),
                            "pred": sy_pred[i].item(),
                            "softmax" : np.round(smsy[i].item(),4)
                        }
                        
                    for i in range(len(eval_data['q_data'].id)):
                        test_task_df[test_task_num]["query"][eval_data['q_data'].id[i].item()]={
                            "ID" : eval_data['q_data'].id[i].item(),
                            "smiles":eval_data['q_data'].smiles[i],
                            "labels":eval_data['q_data'].y[i].item(),
                            "pred": qy_pred[i].item(),
                            "softmax" : np.round(smqy[i].item(),4)
                        }
                    
                y_true = eval_data['q_label']
                class_0_mask = (y_true == 0)
                class_0_accuracy = torch.sum(qy_pred[class_0_mask] == y_true[class_0_mask]).item() / torch.sum(class_0_mask).item()

                # Calculate accuracy for class 1
                class_1_mask = (y_true == 1)
                class_1_accuracy = torch.sum(qy_pred[class_1_mask] == y_true[class_1_mask]).item() / torch.sum(class_1_mask).item()

                # Print the accuracies
                print("Accuracy for class 0: {}, #of0 : {}, #of right {}".format(class_0_accuracy, torch.sum(class_0_mask).item(),torch.sum(qy_pred[class_0_mask] == y_true[class_0_mask]).item()))
                print("Accuracy for class 1: {}, #of1 : {}, #of right {}".format(class_1_accuracy, torch.sum(class_1_mask).item(),torch.sum(qy_pred[class_1_mask] == y_true[class_1_mask]).item()))
                
                auc = auroc(y_score,y_true,num_classes=2).item()
            #summarize        
            auc_scores.append(auc)
            print('Test Epoch:',self.train_epoch,', test for task:', task_id, ', AUC:', round(auc, 4))
            if self.args.save_logs:
                step_results['query_preds'].append(y_score.detach().cpu().numpy())
                step_results['query_labels'].append(y_true.detach().cpu().numpy())
                step_results['task_index'].append(self.test_tasks[task_id].detach().cpu().numpy())
        
        
        #end of all test tasks
        mid_auc = np.median(auc_scores)
        avg_auc = np.mean(auc_scores)
        
        
        
        self.best_auc = max(self.best_auc,avg_auc)
        
        if self.train_epoch == self.args.epochs or self.current_epoch==1 or self.current_epoch%100==0 :
            self.logger.append([self.train_epoch] + auc_scores  +[avg_auc, mid_auc,self.best_auc], verbose=False)
        
        print('Test Epoch:', self.train_epoch, ', AUC_Mid:', round(mid_auc, 4), ', AUC_Avg: ', round(avg_auc, 4),
              ', Best_Avg_AUC: ', round(self.best_auc, 4),)
        
        if self.args.save_logs:
            self.res_logs.append(step_results)
        
        
        torch.cuda.empty_cache()
        return test_task_df, self.best_auc

    def save_model(self):
        save_path = os.path.join(self.trial_path, f"step_{self.train_epoch}.pth")
        torch.save(self.model.module.state_dict(), save_path)
        print(f"Checkpoint saved in {save_path}")
        torch.cuda.empty_cache()

    def save_result_log(self):
        joblib.dump(self.res_logs,self.args.trial_path+'/logs.pkl',compress=6)

    def conclude(self,train_trend):
        df = self.logger.conclude(train_trend)
        self.logger.close()
        print(df)
