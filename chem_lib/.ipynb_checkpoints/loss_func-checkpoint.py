import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
from torch.utils.data import TensorDataset

class Evidence_Classifier(nn.Module):
    def __init__(self, args):
        super(Evidence_Classifier, self).__init__()
        self.device = args.device
        self.args = args
        self.current_epoch = 0
        
    def KL_flat_dirichlet(self, alpha):
        """
        Calculate Kl divergence between a flat/uniform dirichlet distribution and a passed dirichlet distribution
        i.e. KL(dist1||dist2)
        distribution is a flat dirichlet distribution
        :param alpha: The parameters of dist2 (2nd distribution)
        :return: KL divergence
        """
        num_classes = alpha.shape[1]
        beta = torch.ones(alpha.shape, dtype=torch.float32, device=self.device)

        dist1 = dist.Dirichlet(beta)
        dist2 = dist.Dirichlet(alpha)

        kl = dist.kl_divergence(dist1, dist2).reshape(-1, 1)
        return kl

    # A function to calculate the loss based on eqn. 5 of the paper

    def dir_prior_mult_likelihood_loss(self, gt, alpha, current_epoch):
        """
        Calculate the loss based on the dirichlet prior and multinomial likelihoood
        :param gt: The ground truth (one hot vector)
        :param alpha: The prior parameters
        :param current_epoch: For the regularization parameter
        :return: loss
        """
        gt = gt.to(self.device)
        alpha = alpha.to(self.device)

        S = torch.sum(alpha, dim=1, keepdim=True)

        first_part_error = torch.sum(gt * (torch.log(S) - torch.log(alpha)), dim=1, keepdim=True)
        #MSE
        #loglikelihood_err = torch.sum((gt - (alpha / S)) ** 2, dim=1, keepdim=True)
        #loglikelihood_var = torch.sum(
        #    alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
        #)
        #loglikelihood = loglikelihood_err + loglikelihood_var
        #first_part_error = loglikelihood
        
       
        annealing_rate = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(current_epoch / 1000, dtype=torch.float32)
        )

        if self.args.fix_annealing_rate:
            annealing_rate = 1
            # print(annealing_rate)

        alpha_new = (alpha - 1) * (1 - gt) + 1
        kl_err = self.args.kl_scaling_factor * annealing_rate * self.KL_flat_dirichlet(alpha_new)


        #return loss

        dirichlet_strength = torch.sum(alpha, dim=1)
        dirichlet_strength = dirichlet_strength.reshape((-1, 1))

        # Belief
        belief = (alpha - 1) / dirichlet_strength

        inc_belief = belief * (1 - gt)
        inc_belief_error = self.args.kl_scaling_factor * annealing_rate * torch.mean(inc_belief, dim = 1, keepdim=True)

        if self.args.use_kl_error:
            loss = first_part_error + kl_err
            # print("kl using")
        else:
            loss = first_part_error# + inc_belief_error

        return loss


    def calculate_dissonance_from_belief(self, belief):
        num_classes = len(belief)
        Bal_mat = torch.zeros((num_classes, num_classes)).to(self.device)
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                if belief[i] == 0 or belief[j] == 0:
                    Bal_mat[i, j] = 0
                else:
                    Bal_mat[i, j] = 1 - torch.abs(belief[i] - belief[j]) / (belief[i] + belief[j])
                Bal_mat[j, i] = Bal_mat[i, j]
        sum_belief = torch.sum(belief).to(self.device)
        dissonance = 0
        for i in range(num_classes):
            if torch.sum(belief * Bal_mat[i, :]) == 0: continue
            dissonance += belief[i] * torch.sum(belief * Bal_mat[i, :]) / (sum_belief - belief[i])
        return dissonance

    def calculate_dissonance(self, belief):

        dissonance = torch.zeros(belief.shape[0])
        for i in range(len(belief)):
            dissonance[i] = self.calculate_dissonance_from_belief(belief[i])
        return dissonance

    def calculate_dissonance_from_belief_vectorized(self, belief):
        # print("belief shape: ", belief.shape)
        sum_bel_mat = torch.transpose(belief, -2, -1) + belief
        sum_bel_mat[sum_bel_mat == 0] = -500
        # print("sum: ", sum_bel_mat)
        diff_bel_mat = torch.abs(torch.transpose(belief, -2, -1) - belief)
        # print("diff bel mat: ", diff_bel_mat)
        div_diff_sum = torch.div(diff_bel_mat, sum_bel_mat)
        # print("div diff sum up: ", div_diff_sum)

        div_diff_sum[div_diff_sum < -0] = 1
        # print("div diff sum: ", div_diff_sum)
        Bal_mat = 1 - div_diff_sum
        # print("Bal Mat vec: ", Bal_mat)
        # import sys
        # sys.exit()
        num_classes = belief.shape[1]
        Bal_mat[torch.eye(num_classes).byte().bool()] = 0  # torch.zeros((num_classes, num_classes))
        # print("BAL mat: ", Bal_mat)

        sum_belief = torch.sum(belief)

        bel_bal_prod = belief * Bal_mat
        # print("Prod: ", bel_bal_prod)
        sum_bel_bal_prod = torch.sum(bel_bal_prod, dim=1, keepdim=True)
        divisor_belief = sum_belief - belief
        scale_belief = belief / divisor_belief
        scale_belief[divisor_belief == 0] = 1
        each_dis = torch.matmul(scale_belief, sum_bel_bal_prod)

        return torch.squeeze(each_dis)

    def calculate_dissonance_from_belief_vectorized_again(self,belief):
        belief = torch.unsqueeze(belief, dim=1)

        sum_bel_mat = torch.transpose(belief, -2, -1) + belief  # a + b for all a,b in the belief
        diff_bel_mat = torch.abs(torch.transpose(belief, -2, -1) - belief)

        div_diff_sum = torch.div(diff_bel_mat, sum_bel_mat)  # |a-b|/(a+b)

        Bal_mat = 1 - div_diff_sum
        zero_matrix = torch.zeros(sum_bel_mat.shape, dtype=sum_bel_mat.dtype).to(sum_bel_mat.device)
        Bal_mat[sum_bel_mat == zero_matrix] = 0  # remove cases where a=b=0

        diagonal_matrix = torch.ones(Bal_mat.shape[1], Bal_mat.shape[2]).to(sum_bel_mat.device)
        diagonal_matrix.fill_diagonal_(0)  # remove j != k
        Bal_mat = Bal_mat * diagonal_matrix  # The balance matrix

        belief = torch.einsum('bij->bj', belief)
        sum_bel_bal_prod = torch.einsum('bi,bij->bj', belief, Bal_mat)
        sum_belief = torch.sum(belief, dim=1, keepdim=True)
        divisor_belief = sum_belief - belief
        scale_belief = belief / divisor_belief
        scale_belief[divisor_belief == 0] = 1

        each_dis = torch.einsum('bi,bi->b', scale_belief, sum_bel_bal_prod)

        return each_dis


    def calculate_dissonance2(self, belief):
        dissonance = torch.zeros(belief.shape[0])
        for i in range(len(belief)):
            dissonance[i] = self.calculate_dissonance_from_belief_vectorized(belief[i:i + 1, :])
            # break
        return dissonance

    def calculate_dissonance3(self, belief):
        # print("belief: ", belief.shape)
        dissonance = self.calculate_dissonance_from_belief_vectorized_again(belief)
            # break
        return dissonance

    def calc_loss_vac_bel(self, preds, y, query_set=False):
            """
            Calculate the loss, evidence, vacuity, correct belief, and wrong belief
            Prediction is done on the basis of evidence
            :param preds: the NN predictions
            :param y: the groud truth labels
            :param query_set: whether the query set or support set of ask
            :return: loss, vacuity, wrong_belief_vector and cor_belief_vector
            """

            # Make evidence non negative (use softplus)
            evidence = F.softplus(preds)
            # The prior parameters
            alpha = evidence + 1

            dirichlet_strength = torch.sum(alpha, dim=1)
            dirichlet_strength = dirichlet_strength.reshape((-1, 1))
            
            
            # Belief
            belief = evidence / dirichlet_strength
            # Total belief
            sum_belief = torch.sum(belief, dim=1)

            # Vacuity
            vacuity = 1 - sum_belief

            #Dissonance
            dissonance = self.calculate_dissonance3(belief)
            
            # one hot vector for ground truth
            gt = torch.eye(len(y))[y].to(self.device)
            gt = gt[:, :2] ##
            wrong_belief_matrix = belief * (1 - gt)

            wrong_belief_vector = torch.sum(wrong_belief_matrix, dim=1)
            cor_belief_vector = torch.sum(belief * gt, dim=1)

            loss = self.dir_prior_mult_likelihood_loss(gt, alpha, self.current_epoch)
            

            loss = torch.mean(loss)

            return loss, vacuity, wrong_belief_vector, cor_belief_vector, dissonance
        
def compute_similarity(sg_emb, qg_emb):
    similarity = torch.mm(sg_emb,qg_emb.T)
    similarity = torch.nn.functional.normalize(similarity, p=2, dim=1)
    return similarity
def contrastive_loss(sg_emb, qg_emb):
     # Compute denominator of the softmax
    similarity = compute_similarity(sg_emb,qg_emb).to(1)
    temperature = 1
    exp_sim = torch.exp(similarity / temperature)
    denom = torch.sum(exp_sim, dim=1, keepdim=True)
    # Compute numerator of the softmax for positive pairs
    pos_idx = torch.arange(10,20).to(1)
    pos_sim = torch.index_select(similarity, dim=1, index=pos_idx)
    pos_exp_sim = torch.exp(pos_sim / temperature)
    pos_numer = torch.diag(pos_exp_sim)
    # Compute numerator of the softmax for negative pairs
    neg_idx = torch.arange(0, 10).to(1)
    neg_sim = torch.index_select(similarity, dim=1, index=neg_idx)
    neg_exp_sim = torch.sum(torch.exp(neg_sim / temperature), dim=1)
    neg_numer = torch.diag(pos_exp_sim)
    
    # Compute loss
    pos_loss = -torch.log(pos_numer / denom[:10, 0])
    neg_loss = -torch.log(neg_numer / denom[10:, 0])
    loss = torch.mean(pos_loss + neg_loss)
    
    return loss