import numpy as np
from compute_cost import compute_cost

# f(x[i]) = w1*x[i][0] + w2 * x[i][1] + ... + w5*x[5] + b;
# w0 = w0 -eta * Grad( (f(x[0]) - y[0])**2 ) ....
# Grad(f(x[0])) = 1/m * sumi0->4(f(x[i]) - y[i])*x[0][i]



#Normal Gradient Descent Without any optimize
def SubGrad(X, y, w, lamda, b):
    new_W = np.zeros((3,1));
    m = X.shape[0];
    A = X;
    dW = (((np.dot(X,w) + b) - y) / m).reshape(m,1);
    A = A * dW;
    new_W = np.sum(A, axis = 0).reshape((X.shape[1],1)) + w.reshape((X.shape[1],1));
    return new_W;
def Grad_de(X, y, X_test, y_test, w, eta, b, lamda, decay_rate = 0.01):
    cost_history = [];
    new_W = w;
    Fake_W = np.zeros((X.shape[1],1));
    S_corrected_w = np.zeros((X.shape[1],1));
    new_B = 0;
    Fake_B = 0;
    S_corrected_b = 0;
    W_prev = Fake_W;
    m = len(X);
    iter = 1;
    count = 0;
    new_eta = eta;
    m = X.shape[0];
    for iter in range(0,500):
        new_W = SubGrad(X,y,w, lamda, b);
        new_B = 0;
        count = count + 1;
        new_B = np.sum(((np.dot(X,w) + b) - y) / m);
        Fake_W = 0.9 * Fake_W +  0.1 * new_W;
        Fake_B = 0.9 * Fake_B + 0.1 * new_B;
        S_corrected_w = 0.999 * S_corrected_w + 0.001 * (new_W ** 2);
        S_corrected_b = 0.999 * S_corrected_b + 0.001 * (new_B ** 2);
        w = w - (new_eta / (np.sqrt(S_corrected_w) + 1e-8)) * Fake_W;
        b = b - (new_eta / (np.sqrt(S_corrected_b) + 1e-8)) * Fake_B;
        cost_history.append(compute_cost(X_test,y_test,w,b));
        new_eta = eta / (1 - np.floor(count/1000) * decay_rate);
    
    return (w,b, cost_history);
        