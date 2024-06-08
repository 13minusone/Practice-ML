import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

from Gradient_Descent import Grad_de
from compute_cost import compute_cost


# Define the column names 
Data = pd.read_csv("C:\Project\Kaggle\Cali_housing_Price\housing_price_dataset.csv");
y = Data['Price'];
X = Data.drop(columns= ['Price','Neighborhood','Bathrooms']);
Dx = X.to_numpy(dtype= np.float64);
Dy = y.to_numpy(dtype= np.float64);
Max = np.array([-1,-1,-1]);
Min = np.array([1000000,1000000,1000000]);
for i in range(0,len(Dx)):
    for j in range(0,3):
        if Max[j] < Dx[i][j]:
            Max[j] = (Dx[i][j]);
        if Min[j] > Dx[i][j]:
            Min[j] = (Dx[i][j]);
for i in range(0,len(Dx)):
    for j in range(0,3):
        Dx[i][j] = (Dx[i][j] - Min[j]) / (Max[j] - Min[j]);
        if j == 1:
            Dx[i][j] = Dx[i][j] * Dx[i][j];
        if j == 2:
            Dx[i][j] = Dx[i][j] * Dx[i][j] * Dx[i][j];
X_train, X_test, y_train, y_test = train_test_split(Dx, Dy, test_size=0.2, random_state=42);
w = (np.random.rand(X.shape[1])*0.01).reshape(X.shape[1],1);
y_train = y_train.reshape(len(y_train),1);
y_test = y_test.reshape(len(y_test),1);

b = 0;
eta = 2;  
Lamda = 0.7;
###
(w,b, cost_history) = Grad_de(X_train, y_train, X_test, y_test, w, eta, b, Lamda);
###
plt.figure()
plt.plot(range(500), cost_history, label='Cost Function')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Gradient Descent for Linear Regression')
plt.legend()
plt.show();
print(compute_cost(X_test, y_test, w, b));