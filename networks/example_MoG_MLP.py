import torch
from weaver.nn.model.MoG_MLP import *
from weaver.utils.logger import _logger


class MoG_MLPWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mod = MoG_MLP(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }

    def forward(self, points, features, lorentz_vectors, mask,embed = False):
        return self.mod(features, v=lorentz_vectors, mask=mask,embed = embed)


def get_model(data_config, **kwargs):
    print("kwargs:", kwargs)
    

    cfg = dict(
        input_dim=len(data_config.input_dicts['pf_features']),
        num_classes=len(data_config.label_value),
        # network configurations
        pair_input_dim=4,
        use_pre_activation_pair=False,
        pair_embed_dims=[64, 64, 64],
        num_heads=4,
        num_layers=4,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        activation='relu',
        # misc
        trim=True,
        for_inference=False,
    )
    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = MoG_MLPWrapper(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info

    

def get_loss(data_config, **kwargs):
    return ImportanceRegulatedLoss(torch.nn.CrossEntropyLoss())


class ImportanceRegulatedLoss(torch.nn.Module):
    def __init__(self, base_loss, w_1=0.1, w_2=0.1):
        super(ImportanceRegulatedLoss, self).__init__()
        self.base_loss = base_loss
        self.w_1 = w_1
        self.w_2 = w_2

    def forward(self, outputs, targets, router_1, router_2):
        # Base loss (e.g., CrossEntropy)
        b_loss = self.base_loss(outputs, targets)
        
        # Sum of gate values for each router (importance calculation)
        r_1_summed = torch.sum(router_1, dim=0)
        r_2_summed = torch.sum(router_2, dim=0)
        
        # Coefficient of Variation (CV) calculation
        cv_1 = torch.std(r_1_summed) / (torch.mean(r_1_summed) + 1e-8)
        cv_2 = torch.std(r_2_summed) / (torch.mean(r_2_summed) + 1e-8)
        
        # Importance loss component (from equation in image)
        L_importance_1 = self.w_1 * cv_1 ** 2
        L_importance_2 = self.w_2 * cv_2 ** 2
        
        # Total loss (base loss + importance regularization)
        total_loss = b_loss + L_importance_1 + L_importance_2
        
        return total_loss
