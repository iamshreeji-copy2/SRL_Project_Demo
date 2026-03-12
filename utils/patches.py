import io
import pickle
import torch
import sklearn.metrics._dist_metrics

def apply_patches():
    # Monkey patch for older pickle compatibility
    if not hasattr(sklearn.metrics._dist_metrics, 'EuclideanDistance'):
        sklearn.metrics._dist_metrics.EuclideanDistance = sklearn.metrics._dist_metrics.EuclideanDistance64
    if not hasattr(sklearn.metrics._dist_metrics, 'ManhattanDistance'):
        sklearn.metrics._dist_metrics.ManhattanDistance = sklearn.metrics._dist_metrics.ManhattanDistance64

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)
