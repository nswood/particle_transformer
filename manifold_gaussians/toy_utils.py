
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

def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Finds the k-nearest neighbors for each point in a batch of point clouds.

    Args:
        x (torch.Tensor): Input tensor of shape (B, N, F), where
                          B is the batch size,
                          N is the number of points,
                          F is the feature dimension.
        k (int): Number of neighbors to find (excluding the point itself).

    Returns:
        torch.Tensor: Indices of the k-nearest neighbors with shape (B, N, k).
    """
    # Compute pairwise Euclidean distances (B x N x N)
    # p=2 indicates Euclidean distance
    distances = torch.cdist(x, x, p=2)
    
    # Sort the distances along the last dimension.
    # The first index for each point will be the point itself (distance 0).
    sorted_indices = distances.argsort(dim=-1)
    
    # Exclude self by taking neighbors from index 1 to k+1.
    neighbor_indices = sorted_indices[..., 1:k+1]
    
    return neighbor_indices
class CrossAttentionPooling(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super(CrossAttentionPooling, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x, padding_mask=None):
        """
        Adapted to handle both 3D and 4D input.

        If x is 3D:
            x: Tensor of shape (seq_len, batch_size, embed_dim)
        If x is 4D:
            x: Tensor of shape (seq_len, batch_size, extra, embed_dim)
        
        padding_mask:
            For 3D input: shape (batch_size, seq_len) with True indicating masked positions.
            For 4D input: shape (batch_size, extra, seq_len)
        
        Returns:
            For 3D input: Tensor of shape (batch_size, embed_dim) after squeezing the query dimension.
            For 4D input: Tensor of shape (batch_size, extra, embed_dim)
        """
        if x.dim() == 3:
            # Use the original behavior.
            seq_len, batch_size, embed_dim = x.shape
            query = self.query.repeat(1, batch_size, 1)
            # Ensure padding_mask is of shape (batch_size, seq_len)
            if padding_mask is not None:
                # Check if any sample has all tokens masked and adjust if necessary.
                all_masked = padding_mask.all(dim=1)
                if all_masked.any():
                    for idx in torch.nonzero(all_masked, as_tuple=False).squeeze(1):
                        padding_mask[idx, 0] = False
            output, _ = self.attention(query, x, x, key_padding_mask=padding_mask)
            return output.squeeze(0)
        elif x.dim() == 4:
            # Assume shape is (seq_len, batch_size, extra, embed_dim)
            seq_len, batch_size, extra, embed_dim = x.shape
            # Reshape x to merge batch and extra dimensions: (seq_len, batch_size*extra, embed_dim)
            x_reshaped = x.view(seq_len, batch_size * extra, embed_dim)
            # Prepare query: repeat the learnable query for each combined batch element.
            query = self.query.repeat(1, batch_size * extra, 1)
            
            # If a padding_mask is provided, we assume it is of shape (batch_size, extra, seq_len).
            # We need to reshape it to (batch_size*extra, seq_len) for the attention module.
            if padding_mask is not None:
                # Permute from (batch_size, extra, seq_len) to (batch_size, seq_len, extra)
                # then reshape to (batch_size*extra, seq_len)
                padding_mask = padding_mask.permute(0, 2, 1).reshape(batch_size * extra, seq_len)
                # Check for samples where all tokens are masked.
                all_masked = padding_mask.all(dim=1)
                if all_masked.any():
                    for idx in torch.nonzero(all_masked, as_tuple=False).squeeze(1):
                        padding_mask[idx, 0] = False
            
            output, _ = self.attention(query, x_reshaped, x_reshaped, key_padding_mask=padding_mask)
            # output shape: (1, batch_size*extra, embed_dim). Reshape it to (batch_size, extra, embed_dim)
            output = output.squeeze(0).view(batch_size, extra, embed_dim)
            return output
        else:
            raise ValueError("Input tensor must be either 3D or 4D.")


class ParticleCLRModel(nn.Module):
    def __init__(self, input_dim, projection_dim, part_geom, part_dim, k=0, learnable=True, model_name='default', local_geom_weighting = False):
        super(ParticleCLRModel, self).__init__()

        self.all_embedders = nn.ModuleList()
        self.all_manifolds = nn.ModuleList()
        self.local_geom_weighting = local_geom_weighting
        if self.local_geom_weighting:
            local_geom_k_size = [5,8,12]
            self.max_k = max(local_geom_k_size)
    
        # Build the manifold and embedder model inside this class
        if model_name == 'test':
            # For test mode, use a simple embedder and a default Stereographic manifold.
            self.man = geoopt.Stereographic(k=k, learnable=learnable)
            if type(part_dim) == tuple:
                part_dim = part_dim[0]
            elif type(part_dim) == str:
                part_dim = int(part_dim)
            if part_dim is None:
                part_dim = 2
            self.embedder_model = nn.Sequential(
                nn.Linear(input_dim, part_dim),
                nn.ReLU(),
                nn.Linear(part_dim, part_dim)
            )
        else:
            # Build PM-MLP embedder model based on the specified geometry.
            print('Building PM-MLP model')
            print('part_geom:', part_geom)
            print('part_dim:', part_dim)
            print('part_curvature_init:', k)
            print('part_curvature_trainable:', learnable)
            
            if ',' in part_dim:
                part_geom = part_geom.split(',')
                
                part_dim = part_dim.split(',')
                part_dim = tuple([int(x) for x in part_dim])
                k = k.split(',')
                k = tuple([float(x) for x in k])
            else:
                part_dim = [int(part_dim)]
                k = [float(k)]
            self.n_man = len(part_geom)
            for i, geom in enumerate(part_geom):
                if geom == 'R':
                    self.all_manifolds.append(geoopt.Euclidean())
                elif geom == 'H':
                    self.all_manifolds.append(geoopt.PoincareBallExact(k=k[i], learnable=learnable))
                elif geom == 'S':
                    self.all_manifolds.append(geoopt.SphereProjectionExact(k=k[i], learnable=learnable))
                elif geom == 'M':
                    print('k', k[i])
                    self.all_manifolds.append(geoopt.StereographicExact(k=k[i], learnable=learnable))
                else:
                    raise ValueError(f"Unsupported part_geom: {geom}")
                self.all_embedders.append(nn.Sequential(
                        ManifoldNNLayer(input_dim, part_dim[i], k[i], learnable, 0, nn.ReLU(), True),
                        ManifoldNNLayer(part_dim[i], part_dim[i], k[i], learnable, 0, None, True)
                    ))
            if self.local_geom_weighting:
                
                self.max_k = max(local_geom_k_size)
                
                self.local_geom_pooling_fn = nn.ModuleList(CrossAttentionPooling(input_dim, 1) for _ in local_geom_k_size)
            

    def forward(self, x):
        # x: shape (B, C, N)
        B, C, N = x.shape
        x = x.permute(0, 2, 1)

        if self.local_geom_weighting is None:
            neighbor_indices = knn(x, k=self.max_k)
        
        outputs = []
        for i in range(len(self.all_manifolds)):
            # print(self.all_manifolds[i].name)
            # print('Manifold:', self.all_manifolds[i].name)
            cur_x = self.all_manifolds[i].proju(self.all_manifolds[i].origin(x.shape), x)
            # print('cur_x', torch.isnan(cur_x).any())
            cur_x = self.all_manifolds[i].expmap0(cur_x, project=True)
            # print('cur_x on man', torch.isnan(cur_x).any())
            # print('Cur_x', cur_x.shape)
            cur_x = self.all_embedders[i](cur_x)
            # print('cur_x post emb', torch.isnan(cur_x).any())
            if isinstance(cur_x, tuple):
                cur_x = cur_x[1]
            if self.all_manifolds[i].name != 'Euclidean':
                agg = self.all_manifolds[i].weighted_midpoint(cur_x, dim=0, reducedim=[1], keepdim=True)
                agg = self.all_manifolds[i].logmap0(agg)
            else:
                agg = torch.mean(cur_x, dim=1, keepdim=True)
            agg = agg.repeat(1, cur_x.shape[1], 1)
            # print('agg', torch.isnan(agg).any())
            cur_x = self.all_manifolds[i].logmap0(cur_x)
            cur_x = torch.cat((cur_x, agg), dim=-1)
            # print('cur_x cat', torch.isnan(cur_x).any())
            outputs.append(cur_x)
            # print('\n')
        particle_emb = torch.cat(outputs, dim=-1)
        return particle_emb


# ---------------------------------------------------------------------------
# Simple KNN function using Euclidean distance.
def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Finds the k-nearest neighbors for each point in a batch of point clouds.
    
    Args:
        x (torch.Tensor): Input tensor of shape (B, N, F)
        k (int): Number of neighbors to find (excluding self)
        
    Returns:
        torch.Tensor: Indices of k-nearest neighbors (B, N, k)
    """
    distances = torch.cdist(x, x, p=2)
    sorted_indices = distances.argsort(dim=-1)
    neighbor_indices = sorted_indices[..., 1:k+1]
    return neighbor_indices

# ---------------------------------------------------------------------------
# Expert module using the imported ManifoldNNLayer.
class SimpleExpertMLP(nn.Module):
    """
    Processes each expert branch using two sequential manifold layers.
    
    Each layer is built using the imported ManifoldNNLayer with signature:
      (in_features, out_features, k, learnable, dropout, act, use_bias)
    """
    def __init__(self, manifolds, input_dim, output_dim, dropout_rate, activation):
        """
        Args:
            manifolds (ModuleList): List of manifold objects for each expert.
            input_dim (int): Input feature dimension.
            output_dim (int): Output feature dimension for each expert.
            dropout_rate (float): Dropout probability.
            activation (str): Activation function to use (e.g. 'relu').
        """
        super().__init__()
        self.num_experts = len(manifolds)
        self.expert_mlps = nn.ModuleList()
        act = nn.ReLU() if activation=='relu' else None
        use_bias = True
        for m in manifolds:
            if m.name == 'Euclidean':
                mlp = nn.Sequential(
                    nn.Linear(input_dim, output_dim),
                    act,
                    nn.Linear(output_dim, output_dim),
                    act,
                    nn.Linear(output_dim, output_dim),
                    act,
                    nn.Linear(output_dim, output_dim)
                    
                )
            else:
                mlp = nn.Sequential(
                    ManifoldNNLayer(input_dim, output_dim, m, dropout_rate, act, use_bias),
                    ManifoldNNLayer(output_dim, output_dim, m, dropout_rate, act, use_bias),
                    ManifoldNNLayer(output_dim, output_dim, m, dropout_rate, act, use_bias),
                    ManifoldNNLayer(output_dim, output_dim, m, dropout_rate, None, False)
                )

            self.expert_mlps.append(mlp)
    
    def forward(self, x_parts):
        """
        Args:
            x_parts (Tensor): Dense expert inputs of shape (B, N, n_experts, F_in)
            
        Returns:
            proc_parts (Tensor): Processed expert outputs of shape (B, N, n_experts, F_out)
        """
        B, N, n_experts, _ = x_parts.shape
        proc_parts = []
        for k in range(n_experts):
            expert_out = self.expert_mlps[k](x_parts[:, :, k, :])
            proc_parts.append(expert_out)
        proc_parts = torch.stack(proc_parts, dim=2)  # (B, N, n_experts, F_out)
        return proc_parts

# ---------------------------------------------------------------------------
# DenseMoG_MLP with Experts, Local Geometry Weighting, and Weighted Midpoint Concatenation.
class DenseMoG_MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 n_parts=100,
                 local_geom_weighting=True,
                 local_geom_k_size=[5,8,12],
                 part_experts=3,
                 part_expert_curvature_init=[],
                 part_experts_dim=2,
                 shared_expert=False, 
                 shared_expert_ratio=2,
                 particle_feature_agg_method='add+norm',
                 activation='relu',
                 dropout_rate=0.0,
                 learnable=True, 
                 **kwargs):
        super().__init__(**kwargs)
        self.name = 'DenseMoG_MLP'
        self.part_manifolds = nn.ModuleList()
        self.shared_expert = shared_expert
        self.local_geom_weighting = local_geom_weighting
        self.all_k = local_geom_k_size
        self.particle_feature_agg_method = particle_feature_agg_method

        # Local geometry weighting module.
        if self.local_geom_weighting:
            self.max_k = max(local_geom_k_size)
            self.local_geom_pooling_fn = nn.ModuleList(
                nn.Sequential(nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Linear(input_dim, 1))
                for _ in local_geom_k_size
            )

        # Build the manifolds.
        if shared_expert:
            self.part_manifolds.append(geoopt.Euclidean())
            self.part_shared_expert_dim = int(part_experts_dim * shared_expert_ratio)
        if part_expert_curvature_init == []:
            if part_experts % 2 == 1:
                adj = part_experts + 1
                part_expert_curvature_init = np.linspace(-int(adj/2), int(adj/2), adj)[:-1]
            else:
                part_expert_curvature_init = np.linspace(-int(part_experts/2), int(part_experts/2), part_experts)
        for i in range(part_experts):
            if part_expert_curvature_init[i] == 0:
                self.part_manifolds.append(geoopt.Euclidean())
            else:
                self.part_manifolds.append(
                    geoopt.StereographicExact(k=part_expert_curvature_init[i], learnable=learnable)
                )

        self.n_part_man = len(self.part_manifolds)  # All expert branches used.
        print('n_part_man', self.n_part_man )

        # Normalization layers per expert.
        self.part_experts_norms = nn.ModuleList(nn.LayerNorm(input_dim) for _ in range(self.n_part_man))

        # Build the expert module.
        self.part_experts = SimpleExpertMLP(manifolds=self.part_manifolds,
                                            input_dim=input_dim,
                                            output_dim=part_experts_dim,
                                            dropout_rate=dropout_rate,
                                            activation=activation)

        
        
        # Local geometry gating.
        if self.local_geom_weighting:
            self.local_geom_aggregator = nn.ModuleList(CrossAttentionPooling(input_dim, 1) for _ in local_geom_k_size)
            self.local_geom_gating = nn.Sequential(
                nn.Linear(len(self.all_k) * input_dim, self.n_part_man),
                nn.ReLU(),
                nn.Linear(self.n_part_man, self.n_part_man),
                nn.Sigmoid()
            )



    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token', }

    def local_geom_pooling(self, x, v, mask=None, neighbor_indices=None):
        # x: (B, C, N); v: (B, C_v, N); mask: (B, N)
        if neighbor_indices is None:
            x_perm = x.permute(0, 2, 1)  # (B, N, C)
            neighbor_indices = knn(x_perm, k=self.max_k)
        B, C, N = x.shape
        x_perm = x.permute(0, 2, 1)
        pooled_features = []
        for k_i, k in enumerate(self.all_k):
            cur_neighbor_idx = neighbor_indices[:, :, :k]  # (B, N, k)
            batch_indices = torch.arange(B, device=x.device).view(B, 1, 1)
            neighbors = x_perm[batch_indices, cur_neighbor_idx]  # (B, N, k, C)
            aggregated = self.local_geom_aggregator[k_i](neighbors.permute(2,0,1,3))  # (B, N, C)
            pooled_features.append(aggregated)
        combined_local_features = torch.cat(pooled_features, dim=-1)  # (B, N, len(all_k)*C)
        if mask is not None:
            combined_local_features = combined_local_features * mask.unsqueeze(-1).float()
        local_geom_features = combined_local_features  # (N, B, len(all_k)*C)
        local_geom_weights = self.local_geom_gating(local_geom_features)
                                                                                                                                                                           
        
        return local_geom_weights

    def map_onto_manifolds_dense(self, x):
        """
        Map input particles onto each expert's manifold.
        
        Args:
            x: (B, N, C)
            
        Returns:
            Tensor of shape (B, N, n_part_man, C) with each expert branch's mapping.
        """
        B, N, C = x.shape
        x_parts = []
        for k in range(self.n_part_man):
            # print('k',k)
            normed = self.part_experts_norms[k](x)  # (B, N, C)
            # print('normed', normed.shape)
            if self.part_manifolds[k].name == 'Euclidean':
                mapped = normed
            else:
                # print('Manifold', self.part_manifolds[k].name) 
                # print('normed', torch.isinf(normed).any())
                # print('min normed', torch.min(torch.norm(x,dim=-1)))
                # print('max normed', torch.max(torch.norm(x,dim=-1)))

                mapped = self.part_manifolds[k].expmap0(x)
                # print('min mapped', torch.min(torch.norm(mapped,dim=-1)))
                # print('max mapped', torch.max(torch.norm(mapped,dim=-1)))
                
            x_parts.append(mapped)
        x_parts = torch.stack(x_parts, dim=2)  # (B, N, n_part_man, C)
        return x_parts
    
    def map_off_of_manifolds_dense(self, x):
        """
        Map input branches off the manifold (from manifold to tangent space).
        
        Args:
            x_list: A list of length n_experts, where each element is a tensor 
                    of shape (B, N, C) representing the data on that expert's manifold.
                    
        Returns:
            A tensor of shape (B, N, n_experts, C) where each branch has been mapped off
            its manifold. For Euclidean manifolds the branch is returned unchanged.
        """
        
        mapped_parts = []
        B, N, n_experts, _ = x.shape
        proc_parts = []
        for k in range(n_experts):
            x_k = x[:, :, k, :]
            # print('x_k_max', torch.max(x_k))
            # print('x_k_min', torch.min(x_k))
            expert_out = self.part_manifolds[k].logmap0(x_k)
            # print('Expert', self.part_manifolds[k].name)
            # print('expert_max', torch.max(expert_out))
            # print('expert_min', torch.min(expert_out))
            # print('expert_out', torch.isnan(expert_out).any())
            mapped_parts.append(expert_out)
        mapped = torch.stack(mapped_parts, dim=2)
        return mapped

    def scale_PM_embeddings_dense(self, proc_parts, local_geom_weights):
        """
        Multiply each expert branch's output by its corresponding local geometry weight.
        
        Args:
            proc_parts: (B, N, n_part_man, F)
            local_geom_weights: (B, N, n_part_man)
            
        Returns:
            Scaled proc_parts with the same shape.
        """
        weights = local_geom_weights.unsqueeze(-1)  # (B, N, n_part_man, 1)
        return proc_parts * weights

    def combine_expert_with_midpoint(self, proc_parts, local_geom_weights):
        """
        For each expert branch, compute a weighted midpoint of the particle embeddings using the local
        geometry weights, then concatenate this midpoint (broadcast to every particle) to each particle's
        embedding.
        
        Args:
            proc_parts: (B, N, n_part_man, F)
            local_geom_weights: (B, N, n_part_man)
            
        Returns:
            Tensor of shape (B, N, n_part_man * 2 * F) obtained by concatenating, for each expert branch, the
            per-particle embedding and the branch's weighted midpoint.
        """
        eps = 1e-8
        B, N, n_experts, F = proc_parts.shape
        combined_experts = []
        for k in range(n_experts):
            # Extract expert branch k: (B, N, F)
            expert_emb = proc_parts[:, :, k, :]
            # Extract local weights for branch k: (B, N, 1)
            weights = local_geom_weights[:, :, k:k+1]
            # Compute weighted sum over particles: (B, 1, F)
            weighted_sum = (expert_emb * weights).sum(dim=1, keepdim=True)
            # Sum weights over particles: (B, 1, 1)
            weights_sum = weights.sum(dim=1, keepdim=True)
            # Weighted midpoint (avoid division by zero)
            midpoint = weighted_sum / (weights_sum + eps)
            # Broadcast midpoint to all particles: (B, N, F)
            midpoint_broadcast = midpoint.expand(-1, N, -1)
            # Concatenate the original embedding with the broadcast midpoint along feature dimension.
            # combined = torch.cat([expert_emb, midpoint_broadcast], dim=-1)  # (B, N, 2F)
            # combined_experts.append(combined)
            # combined = torch.cat([expert_emb, midpoint_broadcast], dim=-1)  # (B, N, 2F)
            combined_experts.append(expert_emb)
        combined_experts = torch.cat(combined_experts, dim=-1)  # (B, N, n_experts*F)
        return combined_experts

    def forward(self, x, v=None, neighbor_indices=None, mask=None, uu=None, uu_idx=None, embed=False):
        """
        Forward pass.
        
        Expected input x: (C, N, B) --> permuted to (B, N, C)
        Returns:
            A tuple containing:
              - Combined expert embeddings with weighted midpoints concatenated, shape (B, N, n_part_man*2*F)
              - Router weights (B, n_part_man)
              - Processed expert parts (B, N, n_part_man, F)
              - The list of expert manifolds.
        """
        # Permute input from (C, N, B) to (B, N, C)
        x = x.permute(2, 1, 0)
        B, N, C = x.shape
        

    
        # Local geometry weighting.
        if self.local_geom_weighting:
            if neighbor_indices is None:
                neighbor_indices = knn(x, k=self.max_k)
            local_geom_weights = self.local_geom_pooling(x.permute(0, 2, 1), v, mask, neighbor_indices)
            # print('local_geom_weights', torch.isnan(local_geom_weights).any())
        else:
            local_geom_weights = torch.ones(B, N, self.n_part_man, device=x.device)

        # Map input onto all expert manifolds.
        x_parts = self.map_onto_manifolds_dense(x)  # (B, N, n_part_man, C)
        # Process each expert branch.
        proc_parts = self.part_experts(x_parts)  # (B, N, n_part_man, part_experts_dim)
        # Map off of manifolds
        proc_parts = self.map_off_of_manifolds_dense(proc_parts)
        # Scale expert outputs by local geometry weights.
        scaled_proc_parts = self.scale_PM_embeddings_dense(proc_parts, local_geom_weights)  # (B, N, n_part_man, part_experts_dim)
        
        # Combine each expert branch's per-particle embedding with its weighted midpoint.
        combined_expert = self.combine_expert_with_midpoint(scaled_proc_parts, local_geom_weights)
        return combined_expert, local_geom_weights,proc_parts
        # return combined_expert, proc_parts, self.part_manifolds
