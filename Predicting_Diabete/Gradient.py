import numpy as np
from compute_cost import compute_cost

# f(x[i]) = w1*x[i][0] + w2 * x[i][1] + ... + w5*x[5] + b;
# w0 = w0 -eta * Grad( (f(x[0]) - y[0])**2 ) ....
# Grad(f(x[0])) = 1/m * sumi0->4(f(x[i]) - y[i])*x[0][i]


def Sigmoid_Fun(w, X, b):
    Temp = 1;
    Temp = Temp + np.exp( - (np.dot(w, X) + b));
    Temp = 1 / Temp;
    return Temp;
#Normal Gradient Descent Without any optimized

def SubGrad(X, y, w, lamda, b):
    new_W = np.array([0,0,0,0,0, 0], dtype=np.float64);
    m = len(X);
    for i in range(0,6):
        for j in range(0,m):
            xi = X[j];
            new_W[i] = new_W[i] + (Sigmoid_Fun(w,xi,b)/m - y[i]/m)* X[j][i];
        new_W[i] = new_W[i] +  (lamda/m) * w[i];
    return new_W;
def Grad_de(X, y, w, eta, b, lamda):
    new_W = w;
    Fake_W = np.array([0,0,0,0,0, 0], dtype=np.float64);
    new_B = 0;
    Fake_B = 0;
    W_prev = Fake_W;
    m = len(X);
    iter = 1;
    count = 0;
    for iter in range(0,5000):
        new_W = SubGrad(X,y,w, lamda, b);
        new_B = 0;
        count = count + 1;
        W_prev = w
        for j in range(0,m):
            xi = X[j];  
            new_B = new_B + (Sigmoid_Fun(w,xi,b)/m - y[j]/m);
        W_prev = Fake_W;
        Fake_W = 0.9 * Fake_W +  0.1 * new_W;
        Fake_B = 0.9 * Fake_B + 0.1 * new_B;
        w = w - eta * Fake_W;
        b = b - eta * Fake_B;
        if np.linalg.norm(new_W - W_prev) < 1e-4:
            break;
    return (w,b);
        