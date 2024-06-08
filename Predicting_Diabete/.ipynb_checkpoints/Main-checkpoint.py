import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Gradient import Grad_de
from compute_cost import compute_cost
# Define the column names 
Data = pd.read_csv("diabetes.csv");
y = Data['Outcome'];
X = Data.drop(columns= ['Pregnancies','Age','Outcome']);
Dx = X.to_numpy(dtype= np.float64);
Dy = y.to_numpy(dtype= np.float64);
wShape = 6;
print(Dx.shape[0]);
Max = [-1, -1, -1, -1, -1, -1];
Min = [100000, 100000, 100000, 100000, 100000, 100000];
for i in range(0,len(Dx)):
    for j in range(0,wShape):
        if Max[j] < Dx[i][j]:
            Max[j] = (Dx[i][j]);
        if Min[j] > Dx[i][j]:
            Min[j] = (Dx[i][j]);
for i in range(0,len(Dx)):
    for j in range(0,wShape):
        Dx[i][j] = (Dx[i][j] - Min[j]) / (Max[j] - Min[j]);
        # if j == 1:
        #     Dx[i][j] = Dx[i][j] * Dx[i][j];
        # if j == 2:
        #     Dx[i][j] = Dx[i][j] * Dx[i][j] * Dx[i][j];
        # if j == 3:
        #     Dx[i][j] = Dx[i][j] ** 4;
        # if j == 4:
        #     Dx[i][j] = Dx[i][j] ** 5;
X_train, X_test, y_train, y_test = train_test_split(Dx, Dy, test_size=0.05, random_state=42);
print(X_train.shape[0]);
w = np.zeros(wShape, dtype= np.float64);
b = 0;
eta = 0.5;
Lamda = 0;
(w,b) = Grad_de(X_train, y_train, w, eta, b, Lamda);
print(w);
print(b);
print(compute_cost(X_test, y_test, w, b));