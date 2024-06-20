    
import numpy as np
import torch

# Function to recursively find the layer by name
def _get_module_by_name(module, access_string):
    names = access_string.split('.')
    for name in names:
        module = getattr(module, name)
    return module

class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.sq_diff = 0.0

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_size = x.shape[0]
        
        self.n += batch_size
        delta = batch_mean - self.mean
        self.mean += delta * batch_size / self.n
        delta2 = batch_mean - self.mean
        self.sq_diff += delta * delta2 * batch_size

    @property
    def variance(self):
        return self.sq_diff / self.n if self.n > 1 else 0.0

    @property
    def std_dev(self):
        return np.sqrt(self.variance)

class ActivationExtractor:
    def __init__(self):
        self.stats = RunningStats()
        self.accumulated_activations = []

    def hook(self, module, input, output):
        self.accumulated_activations.append(output.cpu().detach().numpy())

    def update_stats(self):
        if self.accumulated_activations:
            activations = np.concatenate(self.accumulated_activations)
            self.stats.update(activations)
            self.accumulated_activations = []