import numpy as np
import pandas as pd
import matplotlib.pyplot as mplib

Data = pd.read_csv("housing_price_dataset.csv");
Data.info();
print(Data.describe());