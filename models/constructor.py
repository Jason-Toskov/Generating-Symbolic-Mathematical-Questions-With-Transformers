from utils import ModelType
from models.transformer import TransformerModel, DiscriminatorModel

import torch

def build_models(env, params):
    """
    Build models.
    """
    if params.model_type == ModelType.BASIC:
        models = {}
        models['gen_encoder'] = TransformerModel(params, env.id2word, is_encoder=True, with_output=False)
        models['gen_decoder'] = TransformerModel(params, env.id2word, is_encoder=False, with_output=True)
        if params.reload_model:
            loaded_data = torch.load(params.reload_model)
            models["gen_encoder"].load_state_dict(loaded_data["gen_enc_weights"])
            models["gen_decoder"].load_state_dict(loaded_data["gen_dec_weights"])
    elif params.model_type == ModelType.TransGAN:
        models = {}
        models['gen_encoder'] = TransformerModel(params, env.id2word, is_encoder=True, with_output=False)
        models['gen_decoder'] = TransformerModel(params, env.id2word, is_encoder=False, with_output=True)
        models['disc'] = DiscriminatorModel(params, env)
        if params.reload_model:
            loaded_data = torch.load(params.reload_model)
            models["gen_encoder"].load_state_dict(loaded_data["gen_enc_weights"])
            models["gen_decoder"].load_state_dict(loaded_data["gen_dec_weights"])
            # models["disc"].load_state_dict(loaded_data["disc_weights"])
            print('Model loaded from',params.reload_model)

    # cuda
    if not params.cpu:
        for v in models.values():
            v.cuda()

    return models