import torch
import numpy as np
import matplotlib.pyplot as plt

from eq_parse_helper import get_math_functions_used, get_steps_used, get_step_names, get_func_names
from utils import ModelType, to_cuda, timeout

@timeout(10)
def integrate_hyp(hyp, env):
    return hyp.integrate(env.local_dict['x'])

@timeout(15)
def timed_get_steps(env, x_out, x_len):
    return get_steps_used(env, x_out, x_len)

class Evaluator():
    def __init__(self, trainer):
        self.trainer = trainer
        self.models = trainer.models
        self.params = trainer.params
        self.env = trainer.env

        self.metrics = {
            'func_match_percent': [],
            'step_match_percent': [],
            'valid_eq_percent': []
        }

    def eval_epoch(self, verbose=False):

        # TODO: This is an abomination, plz fix lol
        invalid_num = 0
        new_num = 0
        old_num = 0
        perc_tok_matched = []
        perc_step_matched = []

        with torch.no_grad():
            for data_type in ['valid', 'test']:
                for task in self.params.tasks:
                    invalid_num, new_num, old_num, perc_tok_matched, perc_step_matched = self.eval_task(data_type, task, invalid_num, new_num, old_num, perc_tok_matched, perc_step_matched, verbose)

        old_eqns_proper = old_num - invalid_num
        print('Summary:')
        print('- Total equations:',new_num + old_num)
        print('- Invalid equations:', invalid_num)
        print('- New equations:', new_num)
        print('- Old equations:', old_eqns_proper)
        print('Func list:',perc_tok_matched)
        print('Step list:',perc_step_matched)
        print('Average %% function matching',np.mean(perc_tok_matched))
        print('Average %% step matching',np.mean(perc_step_matched))

        self.metrics['func_match_percent'].append(np.mean(perc_tok_matched))
        self.metrics['step_match_percent'].append(np.mean(perc_step_matched))
        self.metrics['valid_eq_percent'].append(100*(new_num + old_num - invalid_num)/(new_num + old_num))

        self.plot_accuracies()


    def eval_task(self, data_type, task, invalid_num, new_num, old_num, perc_tok_matched,perc_step_matched, verbose=False):
        # verbose = True
        # iterator
        # TODO: currently is taking 25 eqns, but uses a batch size of 4, so only 7 outputs
        # Fix batch size to be 1 probably 
        ### Should be fixed but i think it's still memory inefficient

        # TODO: filter data file to only include equations with relevant functions
        _, iterator = self.env.create_test_iterator(data_type, task, params=self.params, data_path=self.trainer.data_path)
        eval_size = len(iterator.dataset)

        reached_integral = 0
        reached_end = 0
        for (x1, len1), (x2, len2), (func, func_len), (step_, step_len), (cond, cond_len), _ in iterator:

            if verbose:
                print('\n')
                squeezed_input = x1.squeeze()[1:]
                word_input = [self.env.id2word[wid.item()] for wid in squeezed_input]
                word_input_cropped = word_input[2:-1]
                infix_input = self.env.prefix_to_infix(word_input_cropped)
                input_sp = self.env.infix_to_sympy(infix_input)
                print('- Input:',input_sp)

            # Get functions used 
            func_seq, func_len = get_math_functions_used(self.env, x1)
            if verbose:
                # breakpoint()
                func_print = func_seq.squeeze().tolist()
                if type(func_print) == int:
                    func_print = [func_print] 
                print('- Functions used:', get_func_names(self.env, func_print))
            
            # Get solution technique used
            # TODO: pass this through as an input
            if verbose:
                steps_seq, steps_len = timed_get_steps(self.env, x1, len1)
                steps_print = steps_seq.squeeze().tolist()
                if type(steps_print) == int:
                    steps_print = [steps_print] 

                # print('NOTE: currently not taken as input')
                print('  - Steps used:', get_step_names(steps_print))

            # Take only the zeroth sequence of functions used as input (currently)
            # TODO: do we want to change this? what is the batch size of the test iterators?
            input_seq = cond[:cond_len[0],0].view(-1, 1)
            input_len = torch.LongTensor([cond_len[0]])
            print_seq = input_seq.squeeze().tolist()
            if type(print_seq) == int:
                print_seq = [print_seq]            

            input_seq, input_len = to_cuda(input_seq, input_len)

            # Generate sequences
            test_gen, test_gen_len = self.generate_using_model(input_seq, input_len)

            # Process generated sequences
            # TODO: fix this to use useful functions from original evaluator.py
            ids = test_gen[:,0][1:].tolist()           # decoded token IDs
            tok = [self.env.id2word[wid] for wid in ids]  # convert to prefix
            crop_end = -1 if tok[-1] == '<s>' else None
            if verbose:
                print('- Prefix expression:', tok)

            try:
                tok_crop = tok[2:crop_end]
                hyp = self.env.prefix_to_infix(tok_crop)       # convert to infix
                hyp = self.env.infix_to_sympy(hyp)        # convert to SymPy
                if verbose:
                    pass
                print('- Generated expression:', hyp)
                # breakpoint()

                res = "OK"
                print('integrating!\n')
                reached_integral += 1
                # breakpoint()
                integral = integrate_hyp(hyp, self.env)
                
                # breakpoint()
                if integral is None:
                    # breakpoint()
                    print('timeout!')
                    integral = '____'
                else:
                    # breakpoint()
                    steps_out, steps_out_len = timed_get_steps(self.env, test_gen, test_gen_len)
                    if verbose:
                        # breakpoint()
                        print('- Output functions used:', get_step_names(steps_out))

                # breakpoint()
                if step_len.item() > 2 and integral is not None:
                    step_matched = np.intersect1d(step_[1:step_len.item()-1].cpu(), steps_out)
                    percent_steps_matched_tmp = 100 * len(step_matched) / (step_len.item()-2)
                    perc_step_matched.append(percent_steps_matched_tmp)
                    # breakpoint()
                    if verbose:
                        print('- Matched steps: ', get_step_names(step_matched))
                    print('%% steps matched: %i' %(percent_steps_matched_tmp))
                else:
                    print('Too short! Steps:',step_)                

                print('  Integral solved: ',integral)
                print('')

                # integral = "____"

                in_dataset = self.trainer.find_eq_in_dataset(tok)
                # ' '.join(tok) in self.trainer.train_datapoints
                
                func_out, func_out_len = get_math_functions_used(self.env, torch.LongTensor(ids).unsqueeze(1))
                if verbose:
                    # breakpoint()
                    func_print = func_out.squeeze().tolist()
                    if type(func_print) == int:
                        func_print = [func_print] 
                    print('- Output functions used:', get_func_names(self.env, func_print))

                # breakpoint()
                # TODO: may want to seperate this into checking matches for both functions and steps seperately
                # TODO: currently if there are no functions to matcch nothing special happens, need to account for that
                # breakpoint()
                if func_len.item() > 1:
                    token_matched = np.intersect1d(func[1:func_len.item()+1].cpu(), func_out[1:,:])
                    percent_tokens_matched_tmp = 100 * len(token_matched) / (func_len.item()-1)
                    perc_tok_matched.append(percent_tokens_matched_tmp)
                    if verbose:
                        print('- Matched funcs: ', get_func_names(self.env, token_matched))
                    print('%% tokens matched: %i' %(percent_tokens_matched_tmp))
                else:
                    print('Too short! Funcs:',func)

                reached_end += 1

            except:
                res = "INVALID PREFIX EXPRESSION"
                hyp = tok
                integral = "___"
                in_dataset = True
                invalid_num += 1

            # new_eq = 'Not new' if in_dataset else 'new'
            if in_dataset:
                new_eq = 'Not new'
                old_num += 1
            else:
                new_eq = 'new'
                new_num += 1
            # print result
            print("%s,  %s,  %s" % (res, new_eq, hyp))
            print('  Integral: ',integral)

            if verbose:
                input('Press [ENTER] to continue:')
                print('\n\n\n')
            
        print('reached_integral:',reached_integral,'reached_end:',reached_end)
        # assert reached_integral == reached_end 

        return invalid_num, new_num, old_num, perc_tok_matched, perc_step_matched

    def generate_using_model(self, input_seq, input_len):
        # if self.params.model_type == ModelType.BASIC:
        encoder = self.models['gen_encoder']
        decoder = self.models['gen_decoder']

        encoder.eval()
        decoder.eval()

        encoded = encoder('fwd', x=input_seq, lengths=input_len, causal=False).transpose(0, 1)
        gen_eq, gen_eq_len = decoder.generate(encoded, input_len, max_len=30, 
                                                    sample_temperature=self.params.temperature, 
                                                    top_k_num =self.params.top_k, 
                                                    top_p_num=self.params.top_p, 
                                                    eq_type='Integral')
        
        return gen_eq, gen_eq_len

    def plot_accuracies(self):
        for type, met in self.metrics.items():
            plt.plot(met, label=type)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        plt.legend()
        plt.savefig(self.params.dump_path + 'metrics.png')
        plt.close()


