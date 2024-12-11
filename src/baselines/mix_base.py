from .gnnv3 import gnn
import random
import numpy as np


class mix_base(gnn):
    def uniform(self, k: float=1.0):
        r"""
        Samples from a uniform distribution over [0, k].
        
        Args:
            k (float, optional): The upper bound. (default: :obj:`1.0`)
        
        Returns:
            x (float): A random number in [0, k].
        """
        return random.random() * k


    def beta(self, alpha: float=1.0):
        r"""
        Samples from a beta distribution with alpha
        
        Args:
            alpha (float, optional): Alpha. (default: :obj:`1.0`)
        
        Returns:
            x (float): A random sample.
        """
        return np.random.beta(alpha, alpha)
    