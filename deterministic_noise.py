import torch
import random
import numpy as np


def set_rng_state(seed):
    old_torch_state = torch.get_rng_state()
    old_torch_cuda_state = torch.cuda.get_rng_state()
    old_numpy_state = np.random.get_state()
    old_random_state = random.getstate()

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    return old_torch_state, old_torch_cuda_state, old_numpy_state, old_random_state

def restore_rng_state(states):
    old_torch_state, old_torch_cuda_state, old_numpy_state, old_random_state = states

    torch.set_rng_state(old_torch_state)
    torch.cuda.set_rng_state(old_torch_cuda_state)
    np.random.set_state(old_numpy_state)
    random.setstate(old_random_state)

def get_rng_state():
    old_torch_state = torch.get_rng_state()
    old_torch_cuda_state = torch.cuda.get_rng_state()
    old_numpy_state = np.random.get_state()
    old_random_state = random.getstate()
    return old_torch_state, old_torch_cuda_state, old_numpy_state, old_random_state