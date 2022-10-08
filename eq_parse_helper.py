import torch
import itertools
import numpy as np

from sympy.integrals.manualintegrate import integral_steps

from sympy.integrals.manualintegrate import (
    _manualintegrate, integral_steps, evaluates,
    ConstantRule, ConstantTimesRule, PowerRule, AddRule, URule,
    PartsRule, CyclicPartsRule, TrigRule, ExpRule, ReciprocalRule, ArctanRule,
    AlternativeRule, DontKnowRule, RewriteRule
)

# Keep 0 and 1 for the null token + pad to use in func types
STEP_TYPES = [0,0,AlternativeRule, RewriteRule, ConstantRule, ConstantTimesRule,
              AddRule, PowerRule, TrigRule, ExpRule, ReciprocalRule, URule, 
              PartsRule, CyclicPartsRule, ArctanRule, DontKnowRule, 
]


def get_steps_used(env, equation, eq_length):
    # TODO: only works for first equation right now, may want to generalize to all equations

    # Get ground truth eqn as sympy
    gt_eq = [env.id2word[wid] for wid in equation.transpose(0,1)[0][3:eq_length[0].item()-1].tolist()]
    eq_to_solve = env.infix_to_sympy(env.prefix_to_infix(gt_eq))

    # Get the integral steps
    rule = integral_steps(eq_to_solve, env.local_dict['x'])

    # Parse the expression to get steps as indices of techniques
    steps = get_steps(rule, STEP_TYPES, [])

    # Flatten these steps to get 1 list for each alternate method
    steps_flat = unpack_steps(steps)

    

    # get the method with the simplest solution steps
    try:
        steps_flat.sort(key=max)
        steps_flat_list = list(set(steps_flat[0]))
    except:
        # Fix weird edge case where list has element like [1, 2, 3, [4]]
        steps_flat_fixed = []
        for l in steps_flat:
            l_new = []
            for el in l:
                if isinstance(el, list):
                    if len(el) > 1:
                        print('List length error!')
                        breakpoint()
                    l_new.append(el[0])
                else:
                    l_new.append(el)
            steps_flat_fixed.append(l_new)
        
        steps_flat = steps_flat_fixed
        steps_flat.sort(key=max)
        steps_flat_list = list(set(steps_flat[0]))

    steps_seq = torch.LongTensor(steps_flat_list)
    steps_len = torch.LongTensor(len(steps_flat_list))

    return steps_seq, steps_len



def get_math_functions_used(env, equation):
    ops_indices = [env.word2id[op] for op in env.una_ops]
    op2input = {s:i+len(STEP_TYPES) for i,s in enumerate(ops_indices)}
    def op_2_input(idx):
            return op2input[idx]
    input_tokens = []
    for eq in equation.transpose(1,0): 
        matched_funcs = np.intersect1d(eq, ops_indices).tolist()
        shifted_funcs = list(map(op_2_input, matched_funcs))
        input_tokens.append([0]+shifted_funcs)

    eq_start_len = torch.LongTensor([len(l) for l in input_tokens])
    eq_start = torch.LongTensor(list(itertools.zip_longest(*input_tokens, fillvalue=1)))

    return eq_start, eq_start_len

def get_func_names(env, l):
    ops_indices = [env.word2id[op] for op in env.una_ops]
    input2op = {**{i+len(STEP_TYPES):s for i,s in enumerate(ops_indices)},**{0:0}}
    def idx_to_func(idx):
        return env.id2word[input2op[idx]]
    
    return list(map(idx_to_func, l))

# Much of this code was adapted from https://github.com/sympy/sympy_gamma/blob/master/app/logic/intsteps.py



def get_step_names(l):
    def idx_to_step(idx):
        return STEP_TYPES[idx].__name__

    return list(map(idx_to_step, l))


def get_steps(rule, step_types, method_set):
    skip = False
    if type(rule) in step_types:
        if isinstance(rule, AlternativeRule):
            skip = True
            # Split the sets here #TODO
            for val in rule.alternatives:
                alt_set = get_steps(val, step_types, [])
                method_set.append(alt_set)
                # print(method_set)
        else:
            rule_idx = step_types.index(type(rule))
            method_set.append([rule_idx])
            # print(step_types[rule_idx])
    else:
        # If rule not in set return DontKnowRule
        return [[len(step_types)-1]]

    if not skip:
        if type(rule) == tuple:
            rule = rule[0]
        for val in rule._asdict().values():
            if isinstance(val, tuple):
                method_set = get_steps(val, step_types, method_set)
            elif isinstance(val, list):
                for i in val: 
                    method_set = get_steps(i, step_types, method_set)

    return method_set


# It works now!!!! (i think?) 
# Should unpack to give one flat list of methods for each alternative solution
def unpack_steps(steps):
    # sort to have elements in ascending length
    steps.sort(key=len)
    append_list = [] 
    output_list = []
    for item in steps:
        # Add all the integers to the 'append list'
        if len(item) == 1:
            append_list += item
        else:
            # Unpack the intenal list
            tmp_lst = unpack_steps(item)
            # Append 'append_list' to each inner list
            for l in tmp_lst:
                output_list.append(l+append_list)

    # If we don't have any large lists (this list was only ints)
    # Then return a list of 1 list
    if not len(output_list):
        output_list.append(append_list)
    
    return output_list



