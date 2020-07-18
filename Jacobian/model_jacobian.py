"""
Functions to generate correlated random matrices and compute their spectra.
"""

import torch
import numpy as np
import math

def random_jacobian(n, d, classes=10, correlation_length=1.0):
	"""Generates a random jacobian of dimension (n, d)
	"""
	J = torch.normal(0.0, 1.0, size=(n, d))
	return J

