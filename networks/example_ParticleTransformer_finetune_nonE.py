import torch
import torch.nn as nn
import math
import os
os.environ['PYTHONPATH'] = '/n/home11/nswood/geoopt'
import geoopt
from weaver.nn.model.ParticleTransformer import ParticleTransformer
from weaver.utils.logger import _logger

'''
Link to the full model implementation:
https://github.com/hqucms/weaver-core/blob/main/weaver/nn/model/ParticleTransformer.py
'''
    

class ParticleTransformerWrapper(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        in_dim = kwargs['embed_dims'][-1]
        fc_params = kwargs.pop('fc_params')
        num_classes = kwargs.pop('num_classes')
        self.for_inference = kwargs['for_inference']
        
        self.jet_manifolds = nn.ModuleList()
        self.jet_manifolds.append(geoopt.PoincareBallExact(c = float(2), learnable = True))
        self.jet_manifolds.append(geoopt.SphereProjectionExact(k = float(2), learnable = True))
        self.n_man_jet = 2
        
        in_dim = kwargs['embed_dims'][-1]
        fcs_1_1 = []
        fcs_1_2 = []
        i = 0
        
        for out_dim, drop_rate in fc_params:
            if i < 2:
                fcs_1_1.append(nn.Sequential(Manifold_Linear(in_dim, out_dim, ball = self.jet_manifolds[0] ), nn.ReLU(), nn.Dropout(drop_rate)))
                jet_dim = out_dim
            else:
                fcs_1_2.append(nn.Sequential(Manifold_Linear(in_dim, out_dim, ball = self.jet_manifolds[0] ), nn.ReLU(), nn.Dropout(drop_rate)))
            i +=1
            in_dim = out_dim
            
        self.fc_1_1 = nn.Sequential(*fcs_1_1)
        self.fc_1_2 = nn.Sequential(*fcs_1_2)
        
        
        in_dim = kwargs['embed_dims'][-1]
        fcs_2_1 = []
        fcs_2_2 = []
        i = 0
        for out_dim, drop_rate in fc_params:
            if i < 2:
                fcs_2_1.append(nn.Sequential(Manifold_Linear(in_dim, out_dim, ball = self.jet_manifolds[1] ), nn.ReLU(), nn.Dropout(drop_rate)))
            else:
                fcs_2_2.append(nn.Sequential(Manifold_Linear(in_dim, out_dim, ball = self.jet_manifolds[1] ), nn.ReLU(), nn.Dropout(drop_rate)))
            i +=1
            in_dim = out_dim
        self.fc_2_1 = nn.Sequential(*fcs_2_1)
        self.fc_2_2 = nn.Sequential(*fcs_2_2)
        
        self.embed_1 = nn.ModuleList([self.fc_1_1,self.fc_2_1])
        self.embed_2 = nn.ModuleList([self.fc_1_2,self.fc_2_2])

        
        self.fc_final = nn.Linear(2*in_dim, num_classes)
        
        jet_att_dim = 32
        
        self.W_jet = nn.ModuleList()
        for i,man in enumerate(self.jet_manifolds):
            self.W_jet.append(nn.Linear(jet_dim, jet_att_dim))
        self.Th_jet = nn.ModuleList()
        for i,man in enumerate(self.jet_manifolds):
            self.Th_jet.append(nn.Linear(jet_att_dim, 1)) 
        
        kwargs['num_classes'] = None
        kwargs['fc_params'] = None
        self.mod = ParticleTransformer(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }

    def forward(self, points, features, lorentz_vectors, mask):
        x_cls = self.mod(features, v=lorentz_vectors, mask=mask)
        
        # Non-Euclidean processing
        man_embed = [man.expmap0(x_cls) for man in self.jet_manifolds]
        man_proc_1 = [self.embed_1[i](man_embed[i]) for i in range(len(self.jet_manifolds))]
        
        #Manifold Cross-Attention
        proc_log = [self.jet_manifolds[i].logmap0(self.W_jet[i](man_proc_1[i])) for i in range(self.n_man_jet)]
        mu = torch.stack(proc_log)
        mu = torch.mean(mu, dim=0)
        inter_att = [self.Th_jet[i](proc_log[i]-mu) for i in range(self.n_man_jet)]

        w_i = nn.Softmax(dim=0)(torch.stack(inter_att,dim =0))
        man_post_att = []
        for i in range(self.n_man_jet):
            man_post_att.append(self.jet_manifolds[i].mobius_scalar_mul(w_i[i], man_proc_1[i]))
        
        man_proc_2 = [self.embed_2[i](man_post_att[i]) for i in range(len(self.jet_manifolds))]
        man_log = [self.jet_manifolds[i].logmap0(man_proc_2[i]) for i in range(len(self.jet_manifolds))]
        
        # Combine PM in tan space
        output = torch.cat(man_log,dim=-1)
        output = self.fc_final(output)
        
        if self.for_inference:
            output = torch.softmax(output, dim=1)
        return output


def get_model(data_config, **kwargs):

    cfg = dict(
        input_dim=len(data_config.input_dicts['pf_features']),
        num_classes=len(data_config.label_value),
        # network configurations
        pair_input_dim=4,
        use_pre_activation_pair=False,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[[64,0.1],[32,0.1],[32,0.1],[32,0.1]],
        activation='gelu',
        # misc
        trim=True,
        for_inference=False,
    )
    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = ParticleTransformerWrapper(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()


class Manifold_Linear(nn.Module):
    def __init__(self, in_features, out_features, ball, bias=True):
        super(Manifold_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ball = ball
        self.weight = geoopt.ManifoldParameter(torch.Tensor(out_features, in_features),manifold=self.ball)
        if bias:
            self.bias = geoopt.ManifoldParameter(torch.Tensor(out_features),manifold=self.ball)
        else:
            self.register_parameter("bias", None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(0.1))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, c=None):

        mv = self.ball.mobius_matvec(self.weight, x)

        if not self.bias is None:
            mv = self.ball.mobius_add(mv, self.bias)
        return self.ball.projx(mv)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}, c={}".format(
            self.in_features, self.out_features, self.bias is not None, self.ball
        )