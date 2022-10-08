from eq_parse_helper import get_math_functions_used, get_func_names, get_steps_used, get_step_names, STEP_TYPES
import torch
from tqdm import tqdm
from utils import timeout

@timeout(10)
def timed_get_steps(env, x_out, x_len):
    return get_steps_used(env, x_out, x_len)

def condition_dataset(params, env, trainer, evaluator):

    for task in params.tasks:
        for path in trainer.data_path[task]:

            mode = path.split('.')[-1]

            if mode == 'train':
                ds = trainer.ds_dict[task]
                ds.train = False
                ds.random_eval = False
            else:
                ds, _ = env.create_test_iterator(mode, task, params=params, data_path=trainer.data_path)
                ds.random_eval = False

            

            with open('../'+task+'_conditioned.'+mode, mode='w', encoding='utf-8') as f:
                with tqdm(total=len(ds.data)) as pbar:
                    dont_know_count = 0
                    for i in range(len(ds.data)):
                        x,y = ds.read_sample(i)
                        (x_out, x_len), (y_out, y_len), nb_ops = get_eq(env, x,y)
                        # print("Eq %i:"%(i),x,y)

                        func_seq, func_len = get_math_functions_used(env, x_out)
                        func_print_seq = func_seq.squeeze().tolist()
                        if type(func_print_seq) == int:
                            func_print_seq = [func_print_seq]
                        # print("Funcs Used:",get_func_names(env, func_print_seq))
                        try:
                            steps_seq, steps_len = timed_get_steps(env, x_out, x_len)
                        except:
                            dont_know_count += 1
                            print('timed out')
                            continue
                        step_print_seq = steps_seq.squeeze().tolist()
                        if type(step_print_seq) == int:
                            step_print_seq = [step_print_seq]
                        # print("Steps Used:",get_step_names(step_print_seq))
                        # TODO: dont include functions with 'DontKnowRule'
                        if len(STEP_TYPES)-1 in step_print_seq:
                            # print('DontKnowRule detected, skipping!')
                            dont_know_count += 1
                            pbar.set_description("DontKnowRule count = %i"%(dont_know_count))
                            pbar.update()
                            continue

                        # TODO: fix integral steps to make any rules not in list 'DontKnowRule'
                            # -> should technically be done but probably best to double check
                        # breakpoint()
                        func_str_list = [str(a) for a in func_print_seq]
                        steps_str_list = [str(a) for a in step_print_seq]

                        out_str = str(i)+'|'
                        for j, seq in enumerate([x, y, func_str_list, steps_str_list]):
                            out_str += ' '.join(seq)
                            if j < 3:
                                out_str += '\t'
                            else:
                                out_str += '\n'

                        f.write(out_str)

                        pbar.set_description("DontKnowRule count: %i"%(dont_know_count))
                        pbar.update()

        print("Task",task,'has',i,'datapoints')

    params.is_dataset_conditioned = True
    params.reload_data = 'prim_ibp,../prim_ibp_conditioned.train,../prim_ibp_conditioned.valid,../prim_ibp_conditioned.test;prim_fwd,../prim_fwd_conditioned.train,../prim_fwd_conditioned.valid,../prim_fwd_conditioned.test'
    params.max_test_size = 25
    if isinstance(params.tasks,list):
        params.tasks = ','.join(params.tasks)

    return params


def get_eq(env, x,y):
        """
        Collate samples into a batch.
        """
        # x, y = zip(*elements)
        nb_ops = [sum(int(word in env.OPERATORS) for word in seq) for seq in x]
        # for i in range(len(x)):
        #     print(env.prefix_to_infix(env.unclean_prefix(x[i])))
        #     print(env.prefix_to_infix(env.unclean_prefix(y[i])))
        #     print("")
        x = [torch.LongTensor([env.word2id[w] for w in x if w in env.word2id])]
        y = [torch.LongTensor([env.word2id[w] for w in y if w in env.word2id])]
        x, x_len = env.batch_sequences(x)
        y, y_len = env.batch_sequences(y)
        return (x, x_len), (y, y_len), torch.LongTensor(nb_ops)
        # return x,y
