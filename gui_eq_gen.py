import tkinter as tk
from tkinter import CENTER
from tkinter.messagebox import showerror
from turtle import bgcolor

import numpy as np
import sympy as sp
from PIL import Image, ImageTk
from io import BytesIO
from enum import Enum
import warnings
import torch
import math

import utils
from utils import ModelType, to_cuda, timeout
from param_handler import get_parser, set_params
from envs import build_env
from models.constructor import build_models
from eq_parse_helper import STEP_TYPES, get_math_functions_used, get_steps_used, get_step_names, get_func_names

class AllFrames(Enum):
    HOME = 1
    PROPERTIES = 2
    MODEL = 3
    SOLUTION = 4
    MATCHEDPROP = 5
    METRICS = 6

class CheckboxColour(Enum):
    WHITE = 'white'
    RED = 'red'
    GREEN = 'green'
    YELLOW = 'yellow'

@timeout(5)
def integrate_hyp(hyp, env):
    return hyp.integrate(env.local_dict['x'])

@timeout(15)
def timed_get_steps(env, x_out, x_len):
    return get_steps_used(env, x_out, x_len)

def weight_frame(frame, row_max, col_max):
    for row in range(row_max+1):
        frame.grid_rowconfigure(row, weight=1)
    for col in range(col_max+1):
        frame.grid_columnconfigure(col, weight=1)

# From https://stackoverflow.com/a/59205881
def on_latex(equation, label, colour):
    expr = "$\displaystyle " + equation + "$"

    #This creates a ByteIO stream and saves there the output of sympy.preview
    f = BytesIO()
    # Grey background colour =  self.controller.cget('bg')[1:].upper()
    # White = "FFFFFF"
    the_color = "{" + colour +"}"
    sp.preview(expr, euler = False, preamble = r"\documentclass{standalone}"
            r"\usepackage{pagecolor}"
            r"\usepackage{amsmath}"
            r"\definecolor{graybg}{HTML}" + the_color +
            r"\pagecolor{graybg}"
            r"\begin{document}",
            viewer = "BytesIO", output = "ps", outputbuffer=f)
    f.seek(0)
    #Open the image as if it were a file. This works only for .ps!
    img = Image.open(f)
    #See note at the bottom
    img.load(scale = 10)
    img = img.resize((int(img.size[0]/2),int(img.size[1]/2)),Image.Resampling.BILINEAR)
    photo = ImageTk.PhotoImage(img)
    label.config(image = photo)
    label.image = photo
    f.close()   

class GenerateFrame(tk.Frame):
    def __init__(self, container, controller):
        super().__init__(container)
        self.container = container
        self.controller = controller

        options = {'padx': 10, 'pady': 10}

        self.header_label = tk.Label(
            self,
            text='Integral Equation Generator',
            font = ('',20),
            borderwidth=1,
        )
        self.header_label.grid(row=0, columnspan=2, sticky='n', padx=10, pady=30)

        self.gen_eq_label = tk.Label(self)
        self.gen_eq_label.grid(row=1, columnspan=2, sticky='nsew', **options)
        self.grid_rowconfigure(1, minsize=150)
        
        self.strvar = tk.StringVar()
        self.strvar.set(self.controller.current_generated_equation.get())
        on_latex(self.strvar.get(), self.gen_eq_label, self.controller.cget('bg')[1:].upper())

        self.gen_button = tk.Button(
            self,
            text='Generate',
            font = ('',12),
            command = self.generate_equation
        )
        self.gen_button.grid(row=2, columnspan=2, sticky='n', padx=10,pady=(10,35))


        self.left_subhead_label = tk.Label(
            self,
            text = 'Gen. params:',
            font = ('',14),
        )
        self.left_subhead_label.grid(row=3, column=0, sticky='n', padx=(10,50),pady=10)

        self.eq_properties_button = tk.Button(
            self,
            text='Equation properties',
            command=lambda : self.controller.change_frame(AllFrames.PROPERTIES)
        )
        self.eq_properties_button.grid(row=4, column=0, sticky='nsew', padx=(10,50),pady=10)

        self.model_type_button = tk.Button(
            self,
            text='Model type',
            command=lambda : self.controller.change_frame(AllFrames.MODEL)
        )
        self.model_type_button.grid(row=5, column=0, sticky='nsew', padx=(10,50),pady=10)


        self.right_subhead_label = tk.Label(
            self,
            text = 'Output info:',
            font = ('',14),
        )
        self.right_subhead_label.grid(row=3, column=1, sticky='n', padx=(50,10),pady=10)

        self.solution_button = tk.Button(
            self,
            text='Solution',
            command=lambda : self.controller.change_frame(AllFrames.SOLUTION)
        )
        self.solution_button.grid(row=4, column=1, sticky='nsew', padx=(50,10),pady=10)

        self.solution_button = tk.Button(
            self,
            text='Matched properties',
            command=lambda : self.controller.change_frame(AllFrames.MATCHEDPROP)
        )
        self.solution_button.grid(row=5, column=1, sticky='nsew', padx=(50,10),pady=10)

        self.metrics_button = tk.Button(
            self,
            text='Metrics',
            command=lambda : self.controller.change_frame(AllFrames.METRICS)
        )
        self.metrics_button.grid(row=6, column=1, sticky='nsew', padx=(50,10),pady=10)

        weight_frame(self, 5, 1)
    
    def generate_using_model(self, input_seq, input_len):
        # if self.params.model_type == ModelType.BASIC:
        encoder = self.controller.models['gen_encoder']
        decoder = self.controller.models['gen_decoder']

        encoder.eval()
        decoder.eval()

        encoded = encoder('fwd', x=input_seq, lengths=input_len, causal=False).transpose(0, 1)
        gen_eq, gen_eq_len = decoder.generate(encoded, input_len, max_len=30, 
                                                    sample_temperature=self.controller.params.temperature, 
                                                    top_k_num =self.controller.params.top_k, 
                                                    top_p_num=self.controller.params.top_p, 
                                                    eq_type='Integral')
        
        return gen_eq, gen_eq_len

    def generate_equation(self):
        prop_frame = self.controller.frames[AllFrames.PROPERTIES]
        # Max number of attempts to generate eqn
        for attempt_num in range(5):
            step_idx_to_use = [0]*prop_frame.step_skip_idx + [i.get() for i in prop_frame.method_selection] + [0]
            step_input_list = np.nonzero(step_idx_to_use)[0].tolist()

            ops_indices = [self.controller.env.word2id[op] for op in self.controller.env.una_ops]
            op2input = {s:i+len(STEP_TYPES) for i,s in enumerate(ops_indices)}
            func_indices = np.nonzero([i.get() for i in prop_frame.func_selection])[0]
            # Dunno if i need this below stuff, maybe could do func_indices + 16?
            ops_to_use = [self.controller.env.una_ops[i] for i in func_indices]
            used_ops_indices = [self.controller.env.word2id[op] for op in ops_to_use]
            def op_2_input(idx):
                return op2input[idx]
            func_input_list = list(map(op_2_input, used_ops_indices))

            input_token_list = [0] + step_input_list + func_input_list + [0]
            input_tensor_len = torch.LongTensor([len(input_token_list)])
            input_tensor = torch.LongTensor(input_token_list).unsqueeze(1)

            input_seq, input_len = to_cuda(input_tensor, input_tensor_len)

            # Generate sequences
            test_gen, test_gen_len = self.generate_using_model(input_seq, input_len)

            ids = test_gen[:,0][1:].tolist()           # decoded token IDs
            tok = [self.controller.env.id2word[wid] for wid in ids]  # convert to prefix
            crop_end = -1 if tok[-1] == '<s>' else None

            try:
                tok_crop = tok[2:crop_end]
                hyp = self.controller.env.prefix_to_infix(tok_crop)       # convert to infix
                hyp = self.controller.env.infix_to_sympy(hyp)        # convert to SymPy

                # Get integral solution
                hyp_integral = integrate_hyp(hyp, self.controller.env)
                self.controller.eq_data['integral'] = sp.latex(hyp_integral)

                # Output colour list
                output_colour = [CheckboxColour.WHITE]*(len(step_idx_to_use)-1-prop_frame.step_skip_idx)*len(ops_indices)

                # Get steps used
                steps_out, steps_out_len = timed_get_steps(self.controller.env, test_gen, test_gen_len)

                if len(step_input_list) > 0:
                    step_matched = np.intersect1d(torch.LongTensor(step_input_list), steps_out)
                    percent_steps_matched = round(100 * len(step_matched) / len(step_input_list))
                    extra_steps = np.setxor1d(step_matched, steps_out)
                    missed_steps = np.setxor1d(step_matched, torch.LongTensor(step_input_list))
                else:
                    step_matched = []
                    extra_steps = []
                    missed_steps = []
                    percent_steps_matched = 0
                
                for idx in step_matched:
                    output_colour[idx-prop_frame.step_skip_idx] = CheckboxColour.GREEN
                for idx in extra_steps:
                    output_colour[idx-prop_frame.step_skip_idx] = CheckboxColour.YELLOW
                for idx in missed_steps:
                    output_colour[idx-prop_frame.step_skip_idx] = CheckboxColour.RED
                
                self.controller.eq_data['step_matched'] = step_matched
                self.controller.eq_data['percent_step_matched'] = percent_steps_matched    

                # Get functions present
                func_out, func_out_len = get_math_functions_used(self.controller.env, torch.LongTensor(ids).unsqueeze(1))
                
                if len(func_input_list) > 0:
                    token_matched = np.intersect1d(torch.LongTensor(func_input_list), func_out[1:,:])
                    percent_tokens_matched = round(100 * len(token_matched) / len(func_input_list))
                    extra_funcs = np.setxor1d(token_matched, func_out[1:,:])
                    missed_funcs = np.setxor1d(token_matched, torch.LongTensor(func_input_list))
                else:
                    token_matched = []
                    extra_funcs = []
                    missed_funcs = []
                    percent_tokens_matched = 0
                self.controller.eq_data['token_matched'] = token_matched
                self.controller.eq_data['percent_token_matched'] = percent_tokens_matched

                for idx in token_matched:
                    output_colour[idx-prop_frame.step_skip_idx-1] = CheckboxColour.GREEN
                for idx in extra_funcs:
                    output_colour[idx-prop_frame.step_skip_idx-1] = CheckboxColour.YELLOW
                for idx in missed_funcs:
                    output_colour[idx-prop_frame.step_skip_idx-1] = CheckboxColour.RED

                self.controller.eq_data['checkbox_colour'] = output_colour

                self.controller.current_generated_equation.set(sp.latex(hyp))
                return
            except:
                continue
    
    def update_outputs(self):
        # breakpoint()
        self.strvar.set(r'\int{}'+self.controller.current_generated_equation.get()+r'\mathrm{d}x')
        on_latex(self.strvar.get(), self.gen_eq_label, self.controller.cget('bg')[1:].upper())

class PropertiesFrame(tk.Frame):
    def __init__(self, container, controller):
        super().__init__(container)
        self.container = container
        self.controller = controller

        options = {'padx': 10, 'pady': 10}

        tk.Label(
            self,
            text = 'Desired Output Properties',
            font = ('',20),
        ).grid(row=0, columnspan=4, sticky='n', padx=10, pady=30)

        tk.Label(
            self,
            text = 'Functions:',
            font = ('',14),
        ).grid(row=1, column=0, columnspan=2, sticky='n', **options)

        tk.Label(
            self,
            text = 'Solution methods:',
            font = ('',14),
        ).grid(row=1, column=2, columnspan=2, sticky='n', padx=(100,10),pady=10)

        # Get the max # of rows for func and step:

        # Func
        self.function_choices = self.controller.env.una_ops
        max_rows_idx_func = math.ceil(len(self.function_choices)/2)

        # Step
        self.step_skip_idx = 4
        self.step_options = STEP_TYPES[self.step_skip_idx:-1]
        self.method_choices = get_step_names([*range(self.step_skip_idx,len(STEP_TYPES)-1)])
        max_rows_idx_step = math.ceil(len(self.function_choices)/2)

        max_rows_idx = max(max_rows_idx_func, max_rows_idx_step)

        self.func_selection = []
        for i, func in enumerate(self.function_choices):
            col_idx =  i // max_rows_idx 
            row_idx = i % max_rows_idx 
            self.func_selection.append(tk.IntVar())
            tk.Checkbutton(self, 
                text=func, 
                variable=self.func_selection[i]
            ).grid(column=col_idx, row=row_idx+2, padx=5, pady=5, sticky='w')

        self.method_selection = []
        for j, method in enumerate(self.method_choices):
            col_idx =  j // max_rows_idx 
            row_idx = j % max_rows_idx 
            padx = (100,5) if col_idx == 0 else 5
            self.method_selection.append(tk.IntVar())
            tk.Checkbutton(self, 
                text=method, 
                variable=self.method_selection[j],
            ).grid(column=2+col_idx, row=row_idx+2, padx=padx, pady=5, sticky='w')

        self.back_button = tk.Button(
            self,
            text='Back',
            command=lambda : self.controller.change_frame(AllFrames.HOME)
        )
        self.back_button.grid(row=max_rows_idx+3, column=3, sticky='e', **options)

        self.clear_button = tk.Button(
            self,
            text='Clear',
            command=self.clear_checkboxes
        )
        self.clear_button.grid(row=max_rows_idx+3, column=0, sticky='w', **options)

        # weight_frame(self, max((i,j))+3, 1)

    def clear_checkboxes(self):
        for ck_val in self.func_selection + self.method_selection:
            ck_val.set(False)
        
class ModelFrame(tk.Frame):
    def __init__(self, container, controller):
        super().__init__(container)
        self.container = container
        self.controller = controller

        options = {'padx': 10, 'pady': 10}

        tk.Label(
            self,
            text = 'Model',
            font = ('',20),
        ).grid(row=0, columnspan=4, sticky='n', padx=10, pady=30)

        tk.Label(
            self,
            text = 'Type:',
            font = ('',14),
        ).grid(row=1, column=0, columnspan=2, sticky='n', padx=(10,50),pady=10)

        tk.Label(
            self,
            text = 'Path:',
            font = ('',14),
        ).grid(row=1, column=2, columnspan=2, sticky='n', padx=(50,10),pady=10)

        self.model_type = tk.IntVar()
        tk.Radiobutton(self, 
            text = 'Basic Transformer', 
            variable = self.model_type,
            value = ModelType.BASIC.value,
        ).grid(row=2, column=0, columnspan=2, sticky='w', padx=(10,50),pady=10)
        tk.Radiobutton(self, 
            text = 'Transformer GAN', 
            variable = self.model_type,
            value = ModelType.TransGAN.value,
        ).grid(row=3, column=0, columnspan=2, sticky='w', padx=(10,50),pady=10)
        self.model_type.set(self.controller.params.model_type.value)

        self.use_custom = tk.IntVar()
        tk.Radiobutton(self, 
            text = 'Default', 
            variable = self.use_custom,
            value = 0,
        ).grid(row=2, column=2, columnspan=2, sticky='w', padx=(50,10),pady=10)
        tk.Radiobutton(self, 
            text = 'Custom:', 
            variable = self.use_custom,
            value = 1,
        ).grid(row=3, column=2, sticky='w', padx=(50,10),pady=10)
        self.use_custom.set(0)

        self.custom_path = tk.StringVar()
        tk.Entry(
            self,
            textvariable=self.custom_path
        ).grid(row=4, column=2, columnspan=2, sticky='w', **options)

        self.load_button = tk.Button(
            self,
            text='Load Model',
            command=self.load_model
        )
        self.load_button.grid(row=5, column=0, columnspan=2, sticky='w', **options)

        self.back_button = tk.Button(
            self,
            text='Back',
            command=lambda : self.controller.change_frame(AllFrames.HOME)
        )
        self.back_button.grid(row=5, column=2, columnspan=2, sticky='e', **options)
    
    def load_model(self):
        self.controller.params.model_type = ModelType(self.model_type.get())
        if self.controller.params.model_type == ModelType.BASIC:
            load_path = 'basic_model_best.pt'
        elif self.controller.params.model_type == ModelType.TransGAN:
            load_path = 'transgan_model_best.pt'
        model_path = self.custom_path.get() if self.use_custom.get() else self.controller.params.dump_path + '/' + load_path

        loaded_data = torch.load(model_path)
        for v in self.controller.models.values():
            v.cpu()
        self.controller.models = build_models(self.controller.env, self.controller.params)
        self.controller.models["gen_encoder"].load_state_dict(loaded_data["gen_enc_weights"])
        self.controller.models["gen_decoder"].load_state_dict(loaded_data["gen_dec_weights"])

        print('Model loaded:',model_path)


class SolutionFrame(tk.Frame):
    def __init__(self, container, controller):
        super().__init__(container)
        self.container = container
        self.controller = controller

        options = {'padx': 10, 'pady': 10}

        tk.Label(
            self,
            text = 'Generated equation:',
            font = ('',20),
        ).grid(row=0, sticky='n', padx=10, pady=30)

        self.gen_eq_label = tk.Label(self)
        self.gen_eq_label.grid(row=1, sticky='n', **options)
        self.grid_rowconfigure(1, minsize=150)
        
        self.gen_eq_var = tk.StringVar()
        self.gen_eq_var.set(self.controller.current_generated_equation.get())
        on_latex(self.gen_eq_var.get(), self.gen_eq_label, self.controller.cget('bg')[1:].upper())

        tk.Label(
            self,
            text = 'Solution:',
            font = ('',20),
        ).grid(row=2, sticky='n', padx=10, pady=30)

        self.sol_eq_label = tk.Label(self)
        self.sol_eq_label.grid(row=3, sticky='n', **options)
        self.grid_rowconfigure(1, minsize=150)
        
        self.sol_eq_var = tk.StringVar()
        self.sol_eq_var.set(self.controller.eq_data['integral'])
        on_latex(self.sol_eq_var.get(), self.sol_eq_label, self.controller.cget('bg')[1:].upper())

        self.back_button = tk.Button(
            self,
            text='Back',
            command=lambda : self.controller.change_frame(AllFrames.HOME)
        )
        self.back_button.grid(row=4, sticky='e', **options)
    
    def update_outputs(self):
        self.gen_eq_var.set(r'\int{}'+self.controller.current_generated_equation.get()+'\mathrm{d}x')
        on_latex(self.gen_eq_var.get(), self.gen_eq_label, self.controller.cget('bg')[1:].upper())

        self.sol_eq_var.set(self.controller.eq_data['integral'])
        on_latex(self.sol_eq_var.get(), self.sol_eq_label, self.controller.cget('bg')[1:].upper())

class MatchedPropertiesFrame(tk.Frame):
    def __init__(self, container, controller):
        super().__init__(container)
        self.container = container
        self.controller = controller

        options = {'padx': 10, 'pady': 10}

        tk.Label(
            self,
            text = 'Matched Properties',
            font = ('',20),
        ).grid(row=0, columnspan=4, sticky='n', padx=10, pady=30)

        tk.Label(
            self,
            text = 'Functions:',
            font = ('',14),
        ).grid(row=1, column=0, columnspan=2, sticky='n', **options)

        tk.Label(
            self,
            text = 'Solution methods:',
            font = ('',14),
        ).grid(row=1, column=2, columnspan=2, sticky='n', padx=(100,10),pady=10)

        # Get the max # of rows for func and step:

        # Func
        self.function_choices = self.controller.env.una_ops
        max_rows_idx_func = math.ceil(len(self.function_choices)/2)

        # Step
        self.step_skip_idx = 4
        self.step_options = STEP_TYPES[self.step_skip_idx:-1]
        self.method_choices = get_step_names([*range(self.step_skip_idx,len(STEP_TYPES)-1)])
        max_rows_idx_step = math.ceil(len(self.function_choices)/2)

        max_rows_idx = max(max_rows_idx_func, max_rows_idx_step)

        self.func_selection = []
        self.func_checkboxes = [0]*len(self.function_choices)
        for i, func in enumerate(self.function_choices):
            col_idx =  i // max_rows_idx 
            row_idx = i % max_rows_idx 
            self.func_selection.append(tk.IntVar())
            self.func_checkboxes[i] = tk.Checkbutton(self, 
                text=func, 
                variable=self.func_selection[i],
                selectcolor='white'
            )
            self.func_checkboxes[i] .grid(column=col_idx, row=row_idx+2, padx=5, pady=5, sticky='w')

        self.method_selection = []
        self.method_checkboxes = [0]*len(self.method_choices)
        for j, method in enumerate(self.method_choices):
            col_idx =  j // max_rows_idx 
            row_idx = j % max_rows_idx 
            padx = (100,5) if col_idx == 0 else 5
            self.method_selection.append(tk.IntVar())
            self.method_checkboxes[j] = tk.Checkbutton(self, 
                text=method, 
                variable=self.method_selection[j],
                selectcolor='white'
            )
            self.method_checkboxes[j] .grid(column=2+col_idx, row=row_idx+2, padx=padx, pady=5, sticky='w')

        self.all_checkboxes = self.method_checkboxes + self.func_checkboxes

        self.back_button = tk.Button(
            self,
            text='Back',
            command=lambda : self.controller.change_frame(AllFrames.HOME)
        )
        self.back_button.grid(row=max_rows_idx+3, column=3, sticky='e', **options)

        # weight_frame(self, max((i,j))+3, 1)
    
    def update_outputs(self):
        for ck, colour in zip(self.all_checkboxes, self.controller.eq_data['checkbox_colour']):
            ck.configure(selectcolor=colour.value)


class MetricsFrame(tk.Frame):
    def __init__(self, container, controller):
        super().__init__(container)
        self.container = container
        self.controller = controller

        options = {'padx': 10, 'pady': 10}

        tk.Label(
            self,
            text = 'Equation Details',
            font = ('',20),
        ).grid(row=0, columnspan=2, sticky='n', padx=10, pady=30)

        self.func_match_header = tk.Label(
            self,
            text = r'% funcs matched:',
            font = ('',14),
        )
        self.func_match_header.grid(row=1, column=0, sticky='n', padx=(10,50),pady=10)

        self.func_match_var = tk.StringVar()
        self.func_match_var.set(self.controller.eq_data['percent_token_matched'])

        self.func_match_label = tk.Label(self, textvariable=self.func_match_var)
        self.func_match_label.grid(row=2, column=0, sticky='n', padx=(10,50),pady=10)


        self.step_match_header = tk.Label(
            self,
            text = r'% steps matched:',
            font = ('',14),
        )
        self.step_match_header.grid(row=1, column=1, sticky='n', padx=(50,10),pady=10)

        self.step_match_var = tk.StringVar()
        self.step_match_var.set(self.controller.eq_data['percent_step_matched'])

        self.step_match_label = tk.Label(self, textvariable=self.step_match_var)
        self.step_match_label.grid(row=2, column=1, sticky='n', padx=(50,10),pady=10)

        self.back_button = tk.Button(
            self,
            text='Back',
            command=lambda : self.controller.change_frame(AllFrames.HOME)
        )
        self.back_button.grid(row=3, column=1, sticky='e', **options)

    def update_outputs(self):
        self.func_match_var.set(self.controller.eq_data['percent_token_matched'])
        self.step_match_var.set(self.controller.eq_data['percent_step_matched'])
        


class Root(tk.Tk):
    def __init__(self, env, params, models):
        super().__init__()

        self.env = env
        self.params = params
        self.models = models

        self.current_generated_equation = tk.StringVar(None, ".")
        self.current_generated_equation.trace('w', self.gen_equation_callback)

        self.eq_data = {
            'integral': '.',
            'step_matched': [],
            'percent_step_matched': 0,
            'token_matched': [],
            'percent_token_matched': 0,
            'checkbox_colour': []
        }

        self.title('Equation Generator')

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.geometry('950x800')
        # self.resizable(False, False)
        container = tk.Frame(self)
        container.grid(column=0, row=0, sticky='n')
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        self.frames[AllFrames.HOME] = GenerateFrame(container=container, controller=self)
        self.frames[AllFrames.PROPERTIES] = PropertiesFrame(container=container, controller=self)
        self.frames[AllFrames.MODEL] = ModelFrame(container=container, controller=self)
        self.frames[AllFrames.SOLUTION] = SolutionFrame(container=container, controller=self)
        self.frames[AllFrames.MATCHEDPROP] = MatchedPropertiesFrame(container=container, controller=self)
        self.frames[AllFrames.METRICS] = MetricsFrame(container=container, controller=self)

        self.current_frame = None

        self.change_frame(AllFrames.HOME)

    def change_frame(self, next_frame_key):
        if self.current_frame is not None:
            self.frames[self.current_frame].grid_remove()
        frame = self.frames[next_frame_key]
        frame.grid(column=0, row=0, padx=5, pady=5, sticky='n')
        # frame.grid_rowconfigure(0, weight=1)
        # frame.grid_columnconfigure(0, weight=1)
        self.current_frame = next_frame_key
        # print(next_frame_key,'raised')

    def gen_equation_callback(self, *args):
        self.frames[AllFrames.HOME].update_outputs()
        self.frames[AllFrames.SOLUTION].update_outputs()
        self.frames[AllFrames.MATCHEDPROP].update_outputs()
        self.frames[AllFrames.METRICS].update_outputs()


def main(params):

    print("GPU available:",torch.cuda.is_available())
    utils.CUDA = not params.cpu
    print("CUDA used:",utils.CUDA)

    params.model_type = ModelType.TransGAN
    load_path = 'transgan_model_best.pt'
    loaded_data = torch.load(params.dump_path + load_path)

    env = build_env(params)
    models = build_models(env, params)
    models["gen_encoder"].load_state_dict(loaded_data["gen_enc_weights"])
    models["gen_decoder"].load_state_dict(loaded_data["gen_dec_weights"])

    app = Root(env, params, models)
    app.mainloop()

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    # Parse input arguments
    parser = get_parser()
    params = parser.parse_args()
    params_set = set_params(params)

    params.max_test_size = 100

    main(params_set)
