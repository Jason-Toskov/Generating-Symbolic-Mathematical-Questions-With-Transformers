import argparse
import os
import errno
import signal
import time
import math
from functools import wraps, partial
from enum import Enum

FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

CUDA = True

BASIC_MODEL_STRINGS = {'basic', 'transformer'}
TRANSGAN_MODEL_STRINGS = {'transgan', 'gan'}

def to_cuda(*args):
    """
    Move tensors to CUDA.
    """
    if not CUDA:
        return [None if x is None else x for x in args]
    return [None if x is None else x.cuda() for x in args]

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")

class ModelType(Enum):
    BASIC = 1
    TransGAN = 2

def model_type_enum(s):
    """
    Parse model type from command line.
    """
    if s.lower() in BASIC_MODEL_STRINGS:
        return ModelType.BASIC
    elif s.lower() in TRANSGAN_MODEL_STRINGS:
        return ModelType.TransGAN
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")

class TimeoutError(BaseException):
    pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):

    def decorator(func):

        def _handle_timeout(repeat_id, signum, frame):
            # logger.warning(f"Catched the signal ({repeat_id}) Setting signal handler {repeat_id + 1}")
            signal.signal(signal.SIGALRM, partial(_handle_timeout, repeat_id + 1))
            signal.alarm(seconds)
            # return None
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            old_signal = signal.signal(signal.SIGALRM, partial(_handle_timeout, 0))
            old_time_left = signal.alarm(seconds)
            assert type(old_time_left) is int and old_time_left >= 0
            if 0 < old_time_left < seconds:  # do not exceed previous timer
                signal.alarm(old_time_left)
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
            finally:
                if old_time_left == 0:
                    signal.alarm(0)
                else:
                    sub = time.time() - start_time
                    signal.signal(signal.SIGALRM, old_signal)
                    signal.alarm(max(0, math.ceil(old_time_left - sub)))
            return result

        return wraps(func)(wrapper)

    return decorator


from torch.autograd import Variable
import torch.nn.functional as F
import torch
def sample_gumbel(shape, eps=1e-20):
    U = to_cuda(torch.rand(shape))[0]
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = to_cuda(torch.zeros_like(y).view(-1, shape[-1]))[0]
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y