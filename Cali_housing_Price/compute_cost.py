import numpy as np
import pandas as pd

def compute_cost(X, y, w, b):
    J = 0;
    m = len(X);
    for i in range(0,100):
        Pred_Pri = b;
        Xi = X[i];
        Pred_Pri = Pred_Pri + np.dot(w, Xi);
        J = J + abs(Pred_Pri - y[i])/100 * abs(Pred_Pri - y[i]);
    return J;