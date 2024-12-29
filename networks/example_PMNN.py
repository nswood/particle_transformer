import torch
from weaver.nn.model.PMNN import *
from weaver.utils.logger import _logger

'''
Link to the full model implementation:
https://github.com/hqucms/weaver-core/blob/main/weaver/nn/model/ParticleTransformer.py
'''


class PMNNWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mod = PMNN(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }

    def forward(self, points, features, lorentz_vectors, mask,embed_parts = False):
        return self.mod(features, v=lorentz_vectors, mask=mask,embed_parts=embed_parts)

def get_model(data_config, **kwargs):
    print("kwargs:", kwargs)
    print("data config:", data_config)

    cfg = dict(
        input_dim=len(data_config.input_dicts['pf_features']),
        num_classes=len(data_config.label_value),
        # network configurations
        use_pre_activation_pair=False,
        activation='relu',
#         n_particles = len(data_config.input_dicts['pf_points']),
        # misc
        trim=True,
        for_inference=False,
    )
    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))
   
    
#     if cfg.embedding_mode:
#         model = PMTransformerEmbedderWrapper(**cfg)

#     else:
    model = PMNNWrapper(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
