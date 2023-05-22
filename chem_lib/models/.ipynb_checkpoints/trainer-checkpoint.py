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
from ..datasets import sample_meta_datasets, sample_test_datasets, MoleculeDataset
from ..utils import Logger
from chem_lib.loss_func import Evidence_Classifier, contrastive_loss
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
        #self.optimizer = torch.optim.SGD(model.parameters() , lr=  0.01)
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
        #criterion은 두가지 BCEloss / Dirichlet_loss
        if self.evidence :
            #self.criterion = dirichlet_loss
            #self.criterion = edl_digamma_loss
            self.criterion1 = nn.CrossEntropyLoss().to(args.device)
            #self.criterion2 = edl_mse_loss
            
            self.criterion2 = self.evidence_classifier.calc_loss_vac_bel
        else :
            self.criterion = nn.CrossEntropyLoss().to(args.device)
        
        self.freeze = args.freeze
        
        
        self.trial_path = args.trial_path
        trial_name = self.dataset + '_' + self.test_dataset + '@' + args.enc_gnn
        print(trial_name)
        logger = Logger(self.trial_path + '/results.txt', title=trial_name)
        log_names = ['Epoch']
        log_names += ['AUC-' + str(t) for t in args.test_tasks]
        log_names += ['AUC-Avg', 'AUC-Mid','AUC-Best']
        log_names += ['unc-' + str(t) for t in args.test_tasks]
        #
        #log_names += ['loss']
        #
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
            
            s_data, q_data, pseudo_data = sample_meta_datasets(dataset, self.dataset, task,self.n_shot_train, self.n_query)
            
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

    def get_loss(self, model, data, label,task_id ,train=True, query=False):
        
        if train :
            if self.evidence :
                #pred: logit
                if query : # outer-loop train
                    pred, node_emb, graph_emb,_ = model(data) #pred:logit
                    evidence = F.softplus(pred/self.temperature)
                    alpha = evidence + 1
                    prob = alpha / torch.sum(alpha.detach(),1,keepdim=True)
                    uncertainty = self.n_way/torch.sum(alpha.detach(),1)
                    #one_task_emb = torch.div(torch.mean(graph_emb,0), 2.0)
                    one_task_emb = torch.mean(graph_emb,0).unsqueeze(0)
                    
                    loss, vacuity, wbv, cbv, dis = self.criterion2(pred, label)
                    #print(vacuity,wbv,cbv,dis)
                    #print(alpha,uncertainty)
                    return loss,prob,graph_emb, vacuity,wbv,cbv,dis, one_task_emb
                
                
                else : #inner-loop train
                    pred, node_emb, graph_emb,task_logit = model(data)
                    #print(pred)
                    prob = F.softmax(pred)
                    #print(prob)
                    loss = self.criterion1(prob,label)
                    #task_feature : graph? node?
                    task_feature = torch.mean(graph_emb,0)
                
                    return loss,pred,task_feature
                

            else :
                pred, node_emb,graph_emb = model(data)
                pred = F.softmax(pred)
                loss = self.criterion(pred, label)
                one_task_emb = torch.div(torch.mean(node_emb,0), 2.0)
                return loss, one_task_emb
        else :
            if self.evidence :
                pred, node_emb,graph_emb,_ = model(data)
                #sum of loss
                evidence = F.softplus(pred/self.temperature)
                #evidence = F.softplus(pred/self.temperature)
                
                alpha = evidence + 1
                prob = alpha / torch.sum(alpha.detach(),1,keepdim=True)
                #belief = evidence / torch.sum(alpha.detach(),1,keepdim=True)
                #uncertainty
                uncertainty = self.n_way/torch.sum(alpha.detach(),1)#vacuity
                #dissonance = self.evidence_classifier.calculate_dissonance3(belief)
                ###실험 query를 테스트에 포함시켜 뭘로 데이터를 나누거나 pred의 중간값으로

                    
                
                return pred, prob, uncertainty
            else :
                
                pred, node_emb,graph_emb = model(data)
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
                #support set 에 대해 진행
                for inner_step in range(self.inner_update_step): #1번(adopt per task)
                    adaptable_weights = self.get_adaptable_weights(model)
                    
                    ###inner loop
                    if self.evidence:
                        loss_adapt,sprob,sgraph_emb = self.get_loss(model, train_data['s_data'],train_data['s_label'],task_id,train=True,query=False)
                    else :
                        loss_adapt,_ = self.get_loss(model, train_data['s_data'],train_data['s_label'],train=True,query=False)

                    model.adapt(loss_adapt, adaptable_weights = adaptable_weights)
                     
                    ####outer loop : query 데이터에 대한 loss
                    if self.evidence:#loss,prob,graph_emb, vacuity,wbv,cbv,dis, one_task_emb(tensor로 들어옴)
                        loss_eval,qprob,qgraph_emb,quncertainty,wbv,cbv,dis, one_task_emb = self.get_loss(model, train_data['q_data'], train_data['q_label'],task_id,train=True,query=True)
                        
                        
                        #For task uncertainty
                        av_q_vac = torch.mean(quncertainty)
                        avg_wbv = torch.mean(wbv)
                        av_q_dis = torch.mean(dis)
                        
                        # For train trend acquisition
                        vac_item = [np.round(i.item(),4) for i in quncertainty]
                        wbv_item = [np.round(i.item(),4) for i in wbv]
                        cbv_item = [np.round(i.item(),4) for i in cbv]
                        dis_item = [np.round(i.item(),4) for i in dis]
                        # mean 추가?
                        task_epoch[task_id] = [vac_item,wbv_item,cbv_item,dis_item]
                        # start = 1.0#self.args.vac_inc_balance
                        # end = 0.5
                        # num_epochs = 50
                        # diff = start - end
                        # lambda_val = 1.0 - diff * min(1,epoch/num_epochs)
                        # one_task_unc = av_q_vac * lambda_val + av_q_dis * (1.0 - lambda_val)
                        
                    else :
                        loss_eval,one_task_emb = self.get_loss(model, train_data['q_data'], train_data['q_label'],train=True,query=True)
                    #loss_eval = loss_eval/(self.n_query)
                    #closs = contrastive_loss(sgraph_emb,qgraph_emb)
                    
                    #task_unc.append(one_task_unc)
                    task_emb.append(one_task_emb)
                    losses_eval.append(loss_eval)
                
                #saving data of each task
                #print(len(train_data['s_data']))
                if self.evidence:
                    train_task_df[task_id]["support"] = {}
                    train_task_df[task_id]["query"] = {}

                    for i in range(self.n_shot_train*2):
                        train_task_df[task_id]["support"][train_data['s_data'].id[i].item()]={
                            "smiles":train_data['s_data'].smiles[i],
                            "labels":train_data['s_data'].y[i].item(),
                            "probability0": np.round(sprob[i][0].item(),4),
                            "probability1": np.round(sprob[i][1].item(),4)
                            #"uncertainty": np.round(suncertainty[i].item(),4)
                        }
                    for i in range(self.n_query*2):
                        train_task_df[task_id]["query"][train_data['q_data'].id[i].item()]={
                            "smiles":train_data['q_data'].smiles[i],
                            "labels":train_data['q_data'].y[i].item(),           
                            "probability0":np.round(qprob[i][0].item(),4),
                            "probability1":np.round(qprob[i][1].item(),4),
                            "uncertainty" :np.round(quncertainty[i].item(),4)}
                
            ##
            
            losses_eval = torch.stack(losses_eval)
            #task_unc = torch.stack(task_unc)
            print(losses_eval)
            #task aware
            task_emb = torch.stack(task_emb)
            task_emb = task_emb.detach()
            task_weight = self.attention(task_emb)
            diff = task_emb - torch.transpose(task_emb,1,0)
            diff_sq = torch.pow(diff, 2)
            task_loss = torch.exp(-(diff_sq.sum(dim=-1) / (10)*2))
            #task_weight = 1
            print(task_loss,task_weight)
            losses_eval = torch.sum(task_loss*task_weight*losses_eval)
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
                ##inner_loop
                for i in range(self.update_step_test):
                    print("update_test: ", i)
                    
                    cur_adapt_data = {'s_data': adapt_data['s_data'], 's_label': adapt_data['s_label']}
                        
                    adaptable_weights = self.get_adaptable_weights(model)
                    
                    loss_adapt,sprob,_ = self.get_loss(model, cur_adapt_data['s_data'], cur_adapt_data['s_label'],task_id,train=True,query=False)
                    #loss_adapt = loss_adapt/(self.n_shot_test*2)
                    model.adapt(loss_adapt, adaptable_weights=adaptable_weights)
                    
                    if i>= self.update_step_test-1:
                        break
            #Meta-Test query step
            model.eval()
            with torch.no_grad():
                #여기서 get_loss -> pred, unc return
                if self.evidence:
                    pred_eval, qprob, quncertainty = self.get_loss(model,eval_data['q_data'],eval_data['q_label'],task_id, train=False)
                     #y_prob,y_score = torch.max(pred_eval,1) #value / indice
                    _,y_pred = torch.max(pred_eval,1) #value / indice
                    #y_score = prob[:,1] # prob of 1
                    y_score = y_pred
                    test_task_df[test_task_num]["support"] = {}
                    test_task_df[test_task_num]["query"] = {}
                    for i in range(self.n_shot_train*2):
                        test_task_df[test_task_num]["support"][cur_adapt_data['s_data'].id[i].item()]={
                            "smiles":cur_adapt_data['s_data'].smiles[i],
                            "labels":cur_adapt_data['s_data'].y[i].item(),
                            "probability0":np.round(sprob[i][0].item(),4),
                            "probability1":np.round(sprob[i][1].item(),4),
                            #"uncertainty":np.round(suncertainty[i].item(),4)
                        }
                    for i in range(len(eval_data['q_data'].id)):
                        test_task_df[test_task_num]["query"][eval_data['q_data'].id[i].item()]={
                            "smiles":eval_data['q_data'].smiles[i],
                            "labels":eval_data['q_data'].y[i].item(),
                            "probability0":np.round(qprob[i][0].item(),4),
                            "probability1":np.round(qprob[i][1].item(),4),
                            "uncertainty":quncertainty[i].item()}
                else:
                    pred_eval, _ = self.get_loss(model,eval_data['q_data'],eval_data['q_label'], train=False)
                    y_score = F.softmax(pred_eval,dim=-1).detach()[:,1]
                y_true = eval_data['q_label']
                auc = auroc(y_score,y_true,num_classes=1).item()
                
                
                if self.evidence:
                    y_prob0 = [np.round(i[0].item(),4) for i in qprob] # if evidence : evidence prob / else :softmax prob
                    y_prob1 = [np.round(i[1].item(),4) for i in qprob]
                    uncertainty = [np.round(i.item(),4) for i in quncertainty] #uncertainty
                    y_true = [np.round(i.item(),4) for i in y_true] #truth label
                else :
                    y_true = [np.round(i.item(),4) for i in y_true] #truth label
                    y_score = [np.round(i.item(),4) for i in y_score]
                    
            auc_scores.append(auc)
            
            if self.evidence:
                unc.append([uncertainty,y_prob0,y_prob1,y_true]) #y_score -> evidence prob
            else : # if not evidence : just softmax probability needed
                
                unc.append([y_score,y_true,0,0])
            print('Test Epoch:',self.train_epoch,', test for task:', task_id, ', AUC:', round(auc, 4))
            
            if self.args.save_logs:
                step_results['query_preds'].append(y_score.detach().cpu().numpy())
                step_results['query_labels'].append(y_true.detach().cpu().numpy())
                step_results['task_index'].append(self.test_tasks[task_id].detach().cpu().numpy())
                if self.evidence:
                    step_results['uncertainty'].append(unc)
                else :
                    unc = unc+[0]*len(self.test_tasks-1)
                    step_results['uncertainty'].append(unc)
        
        #end of all test tasks
        mid_auc = np.median(auc_scores)
        avg_auc = np.mean(auc_scores)
        
        # when testing on jupyter notebook
        if not flag:
            if self.evidence:
                
                return unc,eval_data['q_data']
            else :
                return unc,eval_data['q_data']
        
        
        
        
        self.best_auc = max(self.best_auc,avg_auc)
        
        if self.train_epoch == self.args.epochs or self.current_epoch == 500 or self.current_epoch==1 or self.current_epoch==1000 or self.args.early_stop_count == 15 :
            self.logger.append([self.train_epoch] + auc_scores  +[avg_auc, mid_auc,self.best_auc]+[i for i in unc], verbose=False)
        else :
            temp =[0]*len(self.test_tasks)
            self.logger.append([self.train_epoch] + auc_scores  +[avg_auc, mid_auc,self.best_auc]+temp, verbose=False)
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
