import torch
import torch.nn as nn
import torch.nn.functional as F
# from hyp_utils import *
import sys
import os
sys.path.append('../geoopt')
import geoopt
from geoopt.manifolds.stereographic.math import arsinh, artanh,artan_k
import numpy as np
from itertools import combinations
#import torchsort


class SimCLRLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.1, contrast_mode='all',
                 base_temperature=0.1):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward2(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of samples')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        features = features.squeeze()
        dist = (-features.unsqueeze(0)+features.unsqueeze(0).transpose(0, 1)).norm(dim = -1, p = 2,keepdim = False)
        dist = 1/(1+dist)
        anchor_dot_contrast = torch.div(
            dist,
            self.temperature)
#         anchor_dot_contrast = torch.div(
#             torch.matmul(anchor_feature, contrast_feature.T),
#             self.temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        
        # mask-out self-contrast cases
        batch_size, anchor_count = mask.size()

        # Create a mask of ones
        logits_mask = torch.ones_like(mask)

        # Create an identity matrix and expand it to match batch_size
        identity = torch.eye(anchor_count, device=mask.device).unsqueeze(0).expand(batch_size, -1, -1)

        # If mask is 2D, use the diagonal
        if mask.dim() == 2:
            logits_mask = logits_mask - torch.eye(anchor_count, device=mask.device)

#         logits_mask = torch.scatter(
#             torch.ones_like(mask),
#             1,
#             torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
#             0
#         )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss
    
    def forward(self, x_i, x_j):
        #xdevice = x_i.get_device()
        xdevice = (torch.device('cuda') if x_i.is_cuda else torch.device('cpu'))
        batch_size = x_i.shape[0]
        z_i = F.normalize( x_i, dim=1 )
        z_j = F.normalize( x_j, dim=1 )
        z   = torch.cat( [z_i, z_j], dim=0 )
        similarity_matrix = F.cosine_similarity( z.unsqueeze(1), z.unsqueeze(0), dim=2 )
        sim_ij = torch.diag( similarity_matrix,  batch_size )
        sim_ji = torch.diag( similarity_matrix, -batch_size )
        positives = torch.cat( [sim_ij, sim_ji], dim=0 )
        nominator = torch.exp( positives / self.temperature )
        negatives_mask = ( ~torch.eye( 2*batch_size, 2*batch_size, dtype=bool ) ).float()
        negatives_mask = negatives_mask.to( xdevice )
        denominator = negatives_mask * torch.exp( similarity_matrix / self.temperature )
        loss_partial = -torch.log( nominator / torch.sum( denominator, dim=1 ) )
        loss = torch.sum( loss_partial )/( 2*batch_size )
        return loss

    
class ManifoldSimCLRLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.1, contrast_mode='all',
                 base_temperature=0.1):
        super(ManifoldSimCLRLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None,manifold = None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        features = features.unsqueeze(1).unsqueeze(1)
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            print(labels)
            print(labels.type)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        if manifold is not None and manifold.name != 'Euclidean':  
            features = features.squeeze()
            t1 = manifold.mobius_add(-features.unsqueeze(0),features.unsqueeze(0).transpose(0, 1)).norm(dim = -1, p = 2,keepdim = False)
            dist = 2.0 * geoopt.manifolds.stereographic.math.artan_k(t1,k=manifold.k)
            dist = dist.squeeze()
            dist = 1/(1+dist)
        else:
            dist = (-features.unsqueeze(0)+features.unsqueeze(0).transpose(0, 1)).norm(dim = -1, p = 2,keepdim = False)
            dist = 1/(1+dist)
        anchor_dot_contrast = torch.div(
            dist,
            self.temperature)
        
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss
    
    
    
class CrossManifoldSimCLRLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.1, contrast_mode='all',
                 base_temperature=0.1):
        super(CrossManifoldSimCLRLoss, self).__init__()
        
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.loss_simclr = SimCLRLoss(temperature=temperature, contrast_mode=contrast_mode,
                 base_temperature=base_temperature)
        

    def forward(self, exp_dt_all, labels, mask=None):
        n_man = len(exp_dt_all)-1
        indices = np.linspace(0,n_man,n_man+1, dtype = int) 
        pairs = list(combinations(indices, 2))
        L = 0
        for a,b in pairs:
            rep1 =exp_dt_all[a]
            rep2 = exp_dt_all[b]
            max_size = max(rep1.shape[1],rep2.shape[1])
            if rep1.shape[1] < max_size:
                padding = (0, max_size - rep1.shape[1])  # Pad the second dimension
                rep1 = F.pad(rep1, pad=padding, mode='constant', value=0)
            elif rep2.shape[1] < max_size:
                padding = (0, max_size - rep2.shape[1])  # Pad the second dimension
                rep2 = F.pad(rep2, pad=padding, mode='constant', value=0)
            all_reps = torch.vstack([rep1,rep2])
            all_label = torch.hstack([labels,labels])
            L = L + self.loss_simclr.forward2(all_reps.unsqueeze(1).unsqueeze(1),all_label)
        return L


class PMSimCLR(torch.nn.Module):

    def __init__(self, temperature=0.1, contrast_mode='all',
                 base_temperature=0.1):
        super(PMSimCLR, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.loss_flat = SimCLRLoss(temperature = temperature, base_temperature =base_temperature, contrast_mode = contrast_mode)
        self.loss_man = ManifoldSimCLRLoss(temperature = temperature, base_temperature =base_temperature, contrast_mode = contrast_mode)
        self.loss_cross = CrossManifoldSimCLRLoss(temperature = temperature, base_temperature =base_temperature, contrast_mode = contrast_mode)
        

    def forward(self, x_man,x_tan, label, manifolds):
        n_man = len(manifolds)
        if n_man > 1:
            l = 0
            for i in range(n_man):
                if manifolds[i].name == 'Euclidean':
                    l = l + self.loss_flat.forward2(x_man[i].unsqueeze(1).unsqueeze(1), label)
                else: 
                    l = l + self.loss_man(x_man[i], label, manifold = manifolds[i])
            
            l = l + 0.1 * self.loss_cross.forward(x_tan, labels =label)
        else:
            if manifolds[0].name == 'Euclidean':
                l = self.loss_flat.forward2(x_man[0].unsqueeze(1).unsqueeze(1), label)
            else: 
                l = self.loss_man(x_man[0], label, manifold = manifolds[0])
        return l
