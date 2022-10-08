from utils import ModelType
from models.transformer import TransformerModel, DiscriminatorModel

def build_models(env, params):
    """
    Build models.
    """
    if params.model_type == ModelType.BASIC:
        models = {}
        models['gen_encoder'] = TransformerModel(params, env.id2word, is_encoder=True, with_output=False)
        models['gen_decoder'] = TransformerModel(params, env.id2word, is_encoder=False, with_output=True)
    elif params.model_type == ModelType.TransGAN:
        models = {}
        models['gen_encoder'] = TransformerModel(params, env.id2word, is_encoder=True, with_output=False)
        models['gen_decoder'] = TransformerModel(params, env.id2word, is_encoder=False, with_output=True)
        models['disc'] = DiscriminatorModel(params, env)

    # cuda
    if not params.cpu:
        for v in models.values():
            v.cuda()

    return models