import numpy as np
from compute_cost import compute_cost

# f(x[i]) = w1*x[i][0] + w2 * x[i][1] + ... + w5*x[5] + b;
# w0 = w0 -eta * Grad( (f(x[0]) - y[0])**2 ) ....
# Grad(f(x[0])) = 1/m * sumi0->4(f(x[i]) - y[i])*x[0][i]



#Normal Gradient Descent Without any optimize
def SubGrad(X, y, w, lamda, b):
    new_W = np.array([0,0,0], dtype=np.float64);
    m = len(X);
    for i in range(0,3):
        for j in range(0,m):
            xi = X[j];
            new_W[i] = new_W[i] + ((np.dot(w,xi) + b)/m - y[j]/m) * X[j][i];
        new_W[i] = new_W[i] +  (lamda/m) * w[i];
    return new_W;
def Grad_de(X, y, X_test, y_test, w, eta, b, lamda):
    cost_history = [];
    new_W = w;
    Fake_W = np.array([0,0,0], dtype=np.float64);
    new_B = 0;
    Fake_B = 0;
    W_prev = Fake_W;
    m = len(X);
    iter = 1;
    count = 0;
    for iter in range(0,1000):
        new_W = SubGrad(X,y,w, lamda, b);
        new_B = 0;
        count = count + 1;
        W_prev = w
        for j in range(0,m):
            xi = X[j];  
            new_B = new_B + ((np.dot(w,xi) + b)/m - y[j]/m);
        W_prev = Fake_W;
        Fake_W = 0.9 * Fake_W +  0.1 * new_W;
        Fake_B = 0.9 * Fake_B + 0.1 * new_B;
        w = w - eta * Fake_W;
        b = b - eta * Fake_B;
        cost_history.append(compute_cost(X_test,y_test,w,b));
        # if np.linalg.norm(new_W - W_prev) < (1e-5):
        #     break;
    
    return (w,b, cost_history);
        