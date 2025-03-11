

import torch
from torchmetrics import PermutationInvariantTraining
import glob
import numpy as np
import itertools
from PMSimCLR_loss import ManifoldSimCLRLoss
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append('/n/home11/nswood/weaver-core')
sys.path.append('/n/home11/nswood/MoG')
from weaver.nn.model.PMNN import PMNN
from weaver.nn.model.PM_utils import ManifoldNNLayer
import geoopt


class PermutationInvariantLossVectorized(nn.Module):
    def __init__(self, base_loss_fn, num_classes=4):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        perms = list(itertools.permutations(range(num_classes)))
        self.num_perms = len(perms)
        self.register_buffer('perm_tensor', torch.tensor(perms))

    def forward(self, y_pred, y_true, return_labels = False):
        batch_size, num_points, num_classes = y_pred.shape
        # print('y_pred', y_pred.shape)
        # print('y_true', y_true.shape)
        if num_classes != self.perm_tensor.shape[1]:
            raise ValueError("Mismatch in number of classes between predictions and permutation tensor.")

        y_pred_expanded = y_pred.unsqueeze(1).expand(batch_size, self.num_perms, num_points, num_classes)
        perm_tensor_expanded = self.perm_tensor.view(1, self.num_perms, 1, num_classes).expand(batch_size, self.num_perms, num_points, num_classes)
        y_pred_permuted = torch.gather(y_pred_expanded, dim=3, index=perm_tensor_expanded)
        y_true_expanded = y_true.unsqueeze(1).expand(batch_size, self.num_perms, num_points, num_classes)

        losses = self.base_loss_fn(y_pred_permuted.permute(0, 3, 2, 1), y_true_expanded.permute(0, 3, 2, 1).float())
        losses = torch.sum(losses, dim=1).view(batch_size, self.num_perms)
        best_loss_values, best_perm_idx = losses.min(dim=1)
        loss_value = best_loss_values.mean()

        best_y_pred = y_pred_permuted[torch.arange(batch_size), best_perm_idx, :, :]
        best_preds = best_y_pred.argmax(dim=2)
        y_true_labels = y_true.argmax(dim=2)
        correct = (best_preds == y_true_labels).float().mean()
        if return_labels:
            return loss_value, correct, best_preds, y_true_labels
        else:
            return loss_value, correct


# From the paper GraphMoRE: Mitigating Topological Heterogeneity via Mixture of Riemannian Experts https://arxiv.org/pdf/2412.11085
# The distortion loss is defined as the sum of the squared relative errors between the pairwise distances in the embedding space and the pairwise distances in the graph space.
class DistortionLoss(nn.Module):
    def __init__(self,w_1, **kwargs):
        super().__init__(**kwargs)
        self.w_1 = w_1

    def calculate_embedding_distance_matrix(self, embeddings, manifolds, selected_manifolds,local_topology_weights, particle_mask):
        dist_matrices = []
        for i in range(len(embeddings)):
            cur_dist_matrix = torch.zeros(embeddings[i][0].shape[0], embeddings[i][0].shape[0]).to(selected_manifolds.device)
            # print(f'cur_dist_matrix initialized for batch {i}:', torch.isnan(cur_dist_matrix).any())
            
            cur_local_topology_weights = local_topology_weights[i]
            cur_weight_tensor = cur_local_topology_weights.unsqueeze(1) * cur_local_topology_weights.unsqueeze(0)
            cur_weight_tensor = torch.softmax(cur_weight_tensor, dim=-1)
            
            
            cur_mask  = particle_mask[i]
            cur_matrix_mask = cur_mask.unsqueeze(0) * cur_mask.unsqueeze(1)
            # print(f'cur_matrix_mask for batch {i}:', torch.isnan(cur_matrix_mask).any())
            
            for j, k in enumerate(embeddings[i]):
                cur_particles = embeddings[i][j]
                cur_expert = selected_manifolds[i][j]
                cur_manifold = manifolds[cur_expert]
                
                if cur_manifold.name == 'Euclidean':
                    euclidean_dist_matrix = torch.cdist(cur_particles, cur_particles)**2
                    # print(f'euclidean_dist_matrix for batch {i}, expert {j}:', torch.isnan(euclidean_dist_matrix).any())
                    scaled_distance_matrix = cur_weight_tensor[:,:,cur_expert] * euclidean_dist_matrix
                    # print(f'scaled_distance_matrix (Euclidean) for batch {i}, expert {j}:', torch.isnan(scaled_distance_matrix).any())
                    cur_dist_matrix += scaled_distance_matrix.to(selected_manifolds.device)
                else:
                    manifold_distance_matrix = (cur_manifold.dist_matrix(cur_particles, cur_particles)**2).to(selected_manifolds.device)
                    # print(f'manifold_distance_matrix for batch {i}, expert {j}:', torch.isnan(manifold_distance_matrix).any())
                    scaled_distance_matrix = cur_weight_tensor[:,:,cur_expert] * manifold_distance_matrix
                    # print(f'scaled_distance_matrix (Manifold) for batch {i}, expert {j}:', torch.isnan(scaled_distance_matrix).any())
                    cur_dist_matrix += scaled_distance_matrix.to(selected_manifolds.device)

                cur_dist_matrix = cur_matrix_mask * cur_dist_matrix
                # print(f'cur_dist_matrix after masking for batch {i}:', torch.isnan(cur_dist_matrix).any())
            dist_matrices.append(cur_dist_matrix)
        dist_matrices = torch.stack(dist_matrices)

        return dist_matrices
    
    # No padding mask yet
    def forward(self, local_topology_weights, embeddings,manifolds,particle_mask, selected_manifolds, graph_distance_matrix):

        # Calculating 1/V^2 for zero padded events
        num_particles = particle_mask.sum(1)
        loss_scale_mask = 1 / (num_particles**2)
        loss_scale_mask = loss_scale_mask.unsqueeze(-1).unsqueeze(-1)
        
        # Calculating embedding through product manifold distances
        embedding_distance_matrix = self.calculate_embedding_distance_matrix(embeddings,manifolds,selected_manifolds,local_topology_weights,particle_mask)
        # In future resolve nan values in graph_distance_matrix
        graph_distance_matrix = torch.where(torch.isnan(graph_distance_matrix), embedding_distance_matrix, graph_distance_matrix)

        sum_loss = torch.sum(loss_scale_mask*torch.abs((embedding_distance_matrix / (graph_distance_matrix+10e-10)) ** 2 - 1))

        sum_loss = sum_loss * self.w_1
        B = len(embeddings)

        return sum_loss / B