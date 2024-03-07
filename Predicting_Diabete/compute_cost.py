import numpy as np
import pandas as pd

def Sigmoid_Fun(w, X, b):
    Temp = 1;
    Temp = Temp + np.exp( -(np.dot(w, X) + b) );
    Temp = 1 / Temp;
    return Temp;
def compute_cost(X, y, w, b):
    J = 0;
    S = 0;
    m = len(X);
    for i in range(0,m):
        Pred_Pri = 0;
        Xi = X[i];
        zi = Sigmoid_Fun(w, Xi, b);
        Pred_Pri = -(y[i]*np.log(zi) + (1 - y[i]) * np.log(1 -zi));
        if(abs(zi - y[i]) > 0.5):
            S = S + 1;
        J = J  - Pred_Pri / m;
    print(S, "/",m);
    return J;
