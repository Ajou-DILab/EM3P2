import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
from torch.utils.data import TensorDataset
from sklearn.metrics import auc

# adopted form https://github.com/pandeydeep9/Units-ML-CVPR-22
class Evidence_Classifier(nn.Module):
    def __init__(self, args):
        super(Evidence_Classifier, self).__init__()
        self.device = args.device
        self.args = args
        self.current_epoch = 0
        self.eps = 1e-10
        self.disentangle = True
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
        #logarithm
        first_part_error = torch.sum(gt * (torch.log(S) - torch.log(alpha)), dim=1, keepdim=True)
        #MSE
        # loglikelihood_err = torch.sum((gt - (alpha / S)) ** 2, dim=1, keepdim=True)
        # loglikelihood_var = torch.sum(
        #     alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
        # )
        # loglikelihood = loglikelihood_err + loglikelihood_var
        # first_part_error = loglikelihood
        
       
        annealing_rate = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(current_epoch / 2000, dtype=torch.float32)
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

        if self.args.use_kl_error == 1:
            loss = first_part_error + kl_err
            # print("kl using")
        elif self.args.use_kl_error == 2:
            loss = first_part_error + inc_belief_error
        else :
            loss = first_part_error
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
        sum_bel_mat = torch.transpose(belief, -2, -1) + belief
        sum_bel_mat[sum_bel_mat == 0] = -500
        diff_bel_mat = torch.abs(torch.transpose(belief, -2, -1) - belief)
        div_diff_sum = torch.div(diff_bel_mat, sum_bel_mat)

        div_diff_sum[div_diff_sum < -0] = 1
        Bal_mat = 1 - div_diff_sum
        num_classes = belief.shape[1]
        Bal_mat[torch.eye(num_classes).byte().bool()] = 0  # torch.zeros((num_classes, num_classes))

        sum_belief = torch.sum(belief)

        bel_bal_prod = belief * Bal_mat
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


    def aloss(self,gt,alpha,current_epoch):
        gt = gt.to(self.device)
        alpha = alpha.to(self.device)
        S = torch.sum(alpha, dim=1, keepdim=True)
        pred_scores, pred_cls = torch.max(alpha / S, 1, keepdim=True)
        uncertainty = 2 / S
        annealing_rate = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(current_epoch / 2000, dtype=torch.float32)
        )
        acc_match = torch.reshape(torch.eq(pred_cls, gt.unsqueeze(1)).float(), (-1, 1))
        if self.disentangle:
            acc_uncertain = - torch.log(pred_scores * (1 - uncertainty) + self.eps)
            inacc_certain = - torch.log((1 - pred_scores) * uncertainty + self.eps)
        else:
            acc_uncertain = - pred_scores * torch.log(1 - uncertainty + self.eps)
            inacc_certain = - (1 - pred_scores) * torch.log(uncertainty + self.eps)
        avu_loss = annealing_rate * acc_match * acc_uncertain + (1 - annealing_rate) * (1 - acc_match) * inacc_certain

        return avu_loss
    
    
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
            if self.args.use_cal :
                aloss = self.aloss(y,alpha,self.current_epoch)
                loss = loss+0.1*aloss
            
            
            loss = torch.mean(loss)

            #avloss = aloss(gt,alpha)
            
            return loss, vacuity, wrong_belief_vector, cor_belief_vector, dissonance

        ############################################################
        


#
class AUAvULoss(nn.Module):
    """
    Calculates Accuracy vs Uncertainty Loss of a model.
    The input to this loss is logits from Monte_carlo sampling of the model, true labels,
    and the type of uncertainty to be used [0: predictive uncertainty (default); 
    1: model uncertainty]
    """
    def __init__(self, beta=1):
        super(AUAvULoss, self).__init__()
        self.beta = beta
        self.eps = 1e-10

    def entropy(self, prob):
        return -1 * torch.sum(prob * torch.log(prob + self.eps), dim=-1)

    def expected_entropy(self, mc_preds):
        return torch.mean(self.entropy(mc_preds), dim=0)

    def predictive_uncertainty(self, mc_preds):
        """
        Compute the entropy of the mean of the predictive distribution
        obtained from Monte Carlo sampling.
        """
        return self.entropy(torch.mean(mc_preds, dim=0))

    def model_uncertainty(self, mc_preds):
        """
        Compute the difference between the entropy of the mean of the
        predictive distribution and the mean of the entropy.
        """
        return self.entropy(torch.mean(
            mc_preds, dim=0)) - self.expected_entropy(mc_preds)

    def auc_avu(self, probs, labels, unc):
        """ returns AvU at various uncertainty thresholds"""
        th_list = np.linspace(0, 1, 21)
        umin = torch.min(unc)
        umax = torch.max(unc)
        avu_list = []
        unc_list = []

        #probs = F.softmax(logits, dim=1)
        _, predictions = torch.max(probs, 1)
        condfidences = probs[:,1]
        
        auc_avu = torch.ones(1, device=labels.device)
        auc_avu.requires_grad_(True)

        for t in th_list:
            unc_th = umin + (torch.tensor(t) * (umax - umin))
            n_ac = torch.zeros(
                1,
                device=labels.device)  # number of samples accurate and certain
            n_ic = torch.zeros(1, device=labels.device
                               )  # number of samples inaccurate and certain
            n_au = torch.zeros(1, device=labels.device
                               )  # number of samples accurate and uncertain
            n_iu = torch.zeros(1, device=labels.device
                               )  # number of samples inaccurate and uncertain

            for i in range(len(labels)):
                if ((labels[i].item() == predictions[i].item())
                        and unc[i].item() <= unc_th.item()):
                    """ accurate and certain """
                    n_ac += confidences[i] * (1 - torch.tanh(unc[i]))
                elif ((labels[i].item() == predictions[i].item())
                      and unc[i].item() > unc_th.item()):
                    """ accurate and uncertain """
                    n_au += confidences[i] * torch.tanh(unc[i])
                elif ((labels[i].item() != predictions[i].item())
                      and unc[i].item() <= unc_th.item()):
                    """ inaccurate and certain """
                    n_ic += (1 - confidences[i]) * (1 - torch.tanh(unc[i]))
                elif ((labels[i].item() != predictions[i].item())
                      and unc[i].item() > unc_th.item()):
                    """ inaccurate and uncertain """
                    n_iu += (1 - confidences[i]) * torch.tanh(unc[i])

            AvU = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu + 1e-10)
            avu_list.append(AvU.data.cpu().numpy())
            unc_list.append(unc_th)

        auc_avu = auc(th_list, avu_list)
        return auc_avu

    def accuracy_vs_uncertainty(self, prediction, true_label, uncertainty,
                                optimal_threshold):
        n_ac = torch.zeros(
            1,
            device=true_label.device)  # number of samples accurate and certain
        n_ic = torch.zeros(1, device=true_label.device
                           )  # number of samples inaccurate and certain
        n_au = torch.zeros(1, device=true_label.device
                           )  # number of samples accurate and uncertain
        n_iu = torch.zeros(1, device=true_label.device
                           )  # number of samples inaccurate and uncertain

        avu = torch.ones(1, device=true_label.device)
        avu.requires_grad_(True)

        for i in range(len(true_label)):
            if ((true_label[i].item() == prediction[i].item())
                    and uncertainty[i].item() <= optimal_threshold):
                """ accurate and certain """
                n_ac += 1
            elif ((true_label[i].item() == prediction[i].item())
                  and uncertainty[i].item() > optimal_threshold):
                """ accurate and uncertain """
                n_au += 1
            elif ((true_label[i].item() != prediction[i].item())
                  and uncertainty[i].item() <= optimal_threshold):
                """ inaccurate and certain """
                n_ic += 1
            elif ((true_label[i].item() != prediction[i].item())
                  and uncertainty[i].item() > optimal_threshold):
                """ inaccurate and uncertain """
                n_iu += 1

        print('n_ac: ', n_ac, ' ; n_au: ', n_au, ' ; n_ic: ', n_ic, ' ;n_iu: ',
              n_iu)
        avu = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu)

        return avu

    def forward(self, probs, labels, unc):

        #confidences, predictions = torch.max(probs, 1)
        _, predictions = torch.max(probs, 1)
        confidences = probs[:,1]

        th_list = np.linspace(0, 1, 21)
        umin = torch.min(unc)
        umax = torch.max(unc)
        avu_list = []
        unc_list = []

        #probs = F.softmax(logits, dim=1)
        #confidences, predictions = torch.max(probs, 1)

        auc_avu = torch.ones(1, device=labels.device)
        auc_avu.requires_grad_(True)

        for t in th_list:
            unc_th = umin + (torch.tensor(t, device=labels.device) *
                             (umax - umin))
            n_ac = torch.zeros(
                1,
                device=labels.device)  # number of samples accurate and certain
            n_ic = torch.zeros(1, device=labels.device
                               )  # number of samples inaccurate and certain
            n_au = torch.zeros(1, device=labels.device
                               )  # number of samples accurate and uncertain
            n_iu = torch.zeros(1, device=labels.device
                               )  # number of samples inaccurate and uncertain

            for i in range(len(labels)):
                if ((labels[i].item() == predictions[i].item())
                        and unc[i].item() <= unc_th.item()):
                    """ accurate and certain """
                    n_ac += confidences[i] * (1 - torch.tanh(unc[i]))
                elif ((labels[i].item() == predictions[i].item())
                      and unc[i].item() > unc_th.item()):
                    """ accurate and uncertain """
                    n_au += confidences[i] * torch.tanh(unc[i])
                elif ((labels[i].item() != predictions[i].item())
                      and unc[i].item() <= unc_th.item()):
                    """ inaccurate and certain """
                    n_ic += (1 - confidences[i]) * (1 - torch.tanh(unc[i]))
                elif ((labels[i].item() != predictions[i].item())
                      and unc[i].item() > unc_th.item()):
                    """ inaccurate and uncertain """
                    n_iu += (1 - confidences[i]) * torch.tanh(unc[i])

            AvU = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu + self.eps)
            avu_list.append(AvU)
            unc_list.append(unc_th)
        print(torch.stack(avu_list),unc_list)
        auc_avu = auc(th_list, torch.stack(avu_list))
        avu_loss = -1 * self.beta * torch.log(auc_avu + self.eps)
        return avu_loss, auc_avu


class AvULoss(nn.Module):
    """
    Calculates Accuracy vs Uncertainty Loss of a model.
    The input to this loss is logits from Monte_carlo sampling of the model, true labels,
    and the type of uncertainty to be used [0: predictive uncertainty (default); 
    1: model uncertainty]
    """
    def __init__(self, beta=1):
        super(AvULoss, self).__init__()
        self.beta = beta
        self.eps = 1e-10

    def entropy(self, prob):
        return -1 * torch.sum(prob * torch.log(prob + self.eps), dim=-1)

    def expected_entropy(self, mc_preds):
        return torch.mean(self.entropy(mc_preds), dim=0)

    def predictive_uncertainty(self, mc_preds):
        """
        Compute the entropy of the mean of the predictive distribution
        obtained from Monte Carlo sampling.
        """
        return self.entropy(torch.mean(mc_preds, dim=0))

    def model_uncertainty(self, mc_preds):
        """
        Compute the difference between the entropy of the mean of the
        predictive distribution and the mean of the entropy.
        """
        return self.entropy(torch.mean(
            mc_preds, dim=0)) - self.expected_entropy(mc_preds)

    def accuracy_vs_uncertainty(self, prediction, true_label, uncertainty,
                                optimal_threshold):
        # number of samples accurate and certain
        n_ac = torch.zeros(1, device=true_label.device)
        # number of samples inaccurate and certain
        n_ic = torch.zeros(1, device=true_label.device)
        # number of samples accurate and uncertain
        n_au = torch.zeros(1, device=true_label.device)
        # number of samples inaccurate and uncertain
        n_iu = torch.zeros(1, device=true_label.device)

        avu = torch.ones(1, device=true_label.device)
        avu.requires_grad_(True)

        for i in range(len(true_label)):
            if ((true_label[i].item() == prediction[i].item())
                    and uncertainty[i].item() <= optimal_threshold):
                """ accurate and certain """
                n_ac += 1
            elif ((true_label[i].item() == prediction[i].item())
                  and uncertainty[i].item() > optimal_threshold):
                """ accurate and uncertain """
                n_au += 1
            elif ((true_label[i].item() != prediction[i].item())
                  and uncertainty[i].item() <= optimal_threshold):
                """ inaccurate and certain """
                n_ic += 1
            elif ((true_label[i].item() != prediction[i].item())
                  and uncertainty[i].item() > optimal_threshold):
                """ inaccurate and uncertain """
                n_iu += 1

        print('n_ac: ', n_ac, ' ; n_au: ', n_au, ' ; n_ic: ', n_ic, ' ;n_iu: ',
              n_iu)
        avu = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu)

        return avu

    def forward(self, probs, labels,unc, optimal_uncertainty_threshold):

        #probs = F.softmax(logits, dim=1)
        _, predictions = torch.max(probs, 1)
        confidences = probs[:,1]
        # if type == 0:
        #     unc = self.entropy(probs)
        # else:
        #     unc = self.model_uncertainty(probs)

        unc_th = torch.tensor(optimal_uncertainty_threshold,
                              device=labels.device)

        n_ac = torch.zeros(
            1, device=labels.device)  # number of samples accurate and certain
        n_ic = torch.zeros(
            1,
            device=labels.device)  # number of samples inaccurate and certain
        n_au = torch.zeros(
            1,
            device=labels.device)  # number of samples accurate and uncertain
        n_iu = torch.zeros(
            1,
            device=labels.device)  # number of samples inaccurate and uncertain

        avu = torch.ones(1, device=labels.device)
        avu_loss = torch.zeros(1, device=labels.device)

        for i in range(len(labels)):
            if ((labels[i].item() == predictions[i].item())
                    and unc[i].item() <= unc_th.item()):
                """ accurate and certain """
                n_ac += confidences[i] * (1 - torch.tanh(unc[i]))
            elif ((labels[i].item() == predictions[i].item())
                  and unc[i].item() > unc_th.item()):
                """ accurate and uncertain """
                n_au += confidences[i] * torch.tanh(unc[i])
            elif ((labels[i].item() != predictions[i].item())
                  and unc[i].item() <= unc_th.item()):
                """ inaccurate and certain """
                n_ic += (1 - confidences[i]) * (1 - torch.tanh(unc[i]))
            elif ((labels[i].item() != predictions[i].item())
                  and unc[i].item() > unc_th.item()):
                """ inaccurate and uncertain """
                n_iu += (1 - confidences[i]) * torch.tanh(unc[i])

        avu = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu + self.eps)
        p_ac = (n_ac) / (n_ac + n_ic)
        p_ui = (n_iu) / (n_iu + n_ic)
        #print('Actual AvU: ', self.accuracy_vs_uncertainty(predictions, labels, uncertainty, optimal_threshold))
        avu_loss = -1 * self.beta * torch.log(avu + self.eps)
        return avu_loss


def entropy(prob):
    return -1 * np.sum(prob * np.log(prob + 1e-15), axis=-1)


def predictive_entropy(mc_preds):
    """
    Compute the entropy of the mean of the predictive distribution
    obtained from Monte Carlo sampling during prediction phase.
    """
    return entropy(np.mean(mc_preds, axis=0))


def mutual_information(mc_preds):
    """
    Compute the difference between the entropy of the mean of the
    predictive distribution and the mean of the entropy.
    """
    MI = entropy(np.mean(mc_preds, axis=0)) - np.mean(entropy(mc_preds),
                                                      axis=0)
    return MI


def eval_avu(pred_label, true_label, uncertainty):
    """ returns AvU at various uncertainty thresholds"""
    t_list = np.linspace(0, 1, 21)
    umin = np.amin(uncertainty, axis=0)
    umax = np.amax(uncertainty, axis=0)
    avu_list = []
    unc_list = []
    for t in t_list:
        u_th = umin + (t * (umax - umin))
        n_ac = 0
        n_ic = 0
        n_au = 0
        n_iu = 0
        for i in range(len(true_label)):
            if ((true_label[i] == pred_label[i]) and uncertainty[i] <= u_th):
                n_ac += 1
            elif ((true_label[i] == pred_label[i]) and uncertainty[i] > u_th):
                n_au += 1
            elif ((true_label[i] != pred_label[i]) and uncertainty[i] <= u_th):
                n_ic += 1
            elif ((true_label[i] != pred_label[i]) and uncertainty[i] > u_th):
                n_iu += 1

        AvU = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu + 1e-15)
        avu_list.append(AvU)
        unc_list.append(u_th)
    return np.asarray(avu_list), np.asarray(unc_list)


def accuracy_vs_uncertainty(pred_label, true_label, uncertainty,
                            optimal_threshold):

    n_ac = 0
    n_ic = 0
    n_au = 0
    n_iu = 0
    for i in range(len(true_label)):
        if ((true_label[i] == pred_label[i])
                and uncertainty[i] <= optimal_threshold):
            n_ac += 1
        elif ((true_label[i] == pred_label[i])
              and uncertainty[i] > optimal_threshold):
            n_au += 1
        elif ((true_label[i] != pred_label[i])
              and uncertainty[i] <= optimal_threshold):
            n_ic += 1
        elif ((true_label[i] != pred_label[i])
              and uncertainty[i] > optimal_threshold):
            n_iu += 1

    AvU = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu)
    return AvU


