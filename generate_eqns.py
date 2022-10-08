from unicodedata import category
import torch
import utils

from utils import ModelType
from param_handler import get_parser, set_params
from envs import build_env
from models.constructor import build_models
from evaluator import Evaluator
from trainer import Trainer
import warnings


def main(params):

    print("GPU available:",torch.cuda.is_available())
    utils.CUDA = not params.cpu
    print("CUDA used:",utils.CUDA)

    if params.model_type == ModelType.BASIC:
        load_path = 'basic_model_best.pt'
    elif params.model_type == ModelType.TransGAN:
        load_path = 'transgan_model_best.pt'

    loaded_data = torch.load(params.dump_path + '/' + load_path)
    print(params.model_type,'model loaded!')

    env = build_env(params)

    models = build_models(env, params)
    models["gen_encoder"].load_state_dict(loaded_data["gen_enc_weights"])
    models["gen_decoder"].load_state_dict(loaded_data["gen_dec_weights"])

    trainer = Trainer(models, env, params)
    evaluator = Evaluator(trainer) 

    while True:
        evaluator.eval_epoch(verbose=True)

    

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    # Parse input arguments
    parser = get_parser()
    params = parser.parse_args()
    params_set = set_params(params)

    params.max_test_size = 100

    main(params_set)