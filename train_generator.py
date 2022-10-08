from param_handler import get_parser, set_params
from envs import build_env
from models.constructor import build_models
from  construct_conditioned_dataset import condition_dataset
from evaluator import Evaluator
from trainer import Trainer
from tqdm import tqdm
import numpy as np

import utils
import torch

def main(params):
    print("GPU available:",torch.cuda.is_available())
    utils.CUDA = not params.cpu
    print("CUDA used:",utils.CUDA)

    env = build_env(params)

    print(params.model_type)

    models = build_models(env, params)
    trainer = Trainer(models, env, params)
    evaluator = Evaluator(trainer) 

    if not params.is_dataset_conditioned:
        params = condition_dataset(params, env, trainer, evaluator)
        env = build_env(params)
        models = build_models(env, params)
        trainer = Trainer(models, env, params)
        evaluator = Evaluator(trainer) 

    for n in range(params.max_epoch):
        
        print("============ Starting epoch %i ... ============" % trainer.epoch)
        trainer.n_equations = 0
        with tqdm(total=trainer.epoch_size) as pbar:
            while trainer.n_equations < trainer.epoch_size:

                for task_id in np.random.permutation(len(params.tasks)):
                    task = params.tasks[task_id]
                    trainer.train_step(task)

                    pbar.set_description(trainer.loss_logger.get_description())
                    pbar.update(params.batch_size)
        
        evaluator.eval_epoch()
        trainer.end_epoch()


if __name__ == '__main__':
    # Parse input arguments
    parser = get_parser()
    params = parser.parse_args()
    params_set = set_params(params)

    main(params_set)