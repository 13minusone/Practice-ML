import numpy as np
import pandas as pd

def compute_cost(X, y, w, b):
    J = 0;
    m = len(X);
    J =  np.sum(np.square((np.dot(X,w) + b) - y)) / (2 * m);
    return J;