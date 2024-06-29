import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
import gradio as gr
import plotly.graph_objects as go

from Main import LinearRegression

df = pd.read_csv("C:\Project\Kaggle\Cali_housing_Price\housing_price_dataset.csv");
Data = pd.read_csv("C:\Project\Kaggle\Cali_housing_Price\housing_price_dataset.csv");
y = Data['Price'];
X = Data.drop(columns= ['Price']);
for i in range(X['Neighborhood'].shape[0]):
    if X.loc[i,'Neighborhood'] == 'Suburb':
        X.loc[i,'Neighborhood'] = 2;
    elif X.loc[i,'Neighborhood'] == 'Urban':
        X.loc[i,'Neighborhood'] = 3;
    else:
        X.loc[i,'Neighborhood'] = 1;
Dx = X.to_numpy(dtype= np.float64);
Dy = y.to_numpy(dtype= np.float64);
X_train, X_test, y_train, y_test = train_test_split(Dx, Dy, test_size=0.02, random_state=42);
w = (np.random.rand(X.shape[1])*0.01).reshape(X.shape[1],1);
y_train = y_train.reshape(len(y_train),1);
y_test = y_test.reshape(len(y_test),1);
HousePriceModel = LinearRegression();
HousePriceModel.fit(X_train, y_train);
def HousePrice(SquareFeet, Bedrooms, Bathrooms, Neighborhood, YearBuilt):
    NumNeighborhood = 0;
    if Neighborhood == 'Suburb':
        NumNeighborhood = 2;
    elif Neighborhood == 'Urban':
        NumNeighborhood = 3;
    elif Neighborhood == 'Rural':
        NumNeighborhood = 1;
    else:
        raise gr.Error("Invalid Neighborhood");
    if YearBuilt > 2024 or YearBuilt < 1900:
        raise gr.Error("Invalid Year Built");
    if SquareFeet < 0:
        raise gr.Error("Invalid Square Feet");
    if Bedrooms < 0:
        raise gr.Error("Invalid Bedrooms");
    if Bathrooms < 0:
        raise gr.Error("Invalid Bathrooms");
    X = np.array([SquareFeet, Bedrooms, Bathrooms, NumNeighborhood, YearBuilt]).reshape(1,5);
    YPredict = HousePriceModel.predict(X);
    #### Filter the data
    filter_df = df[(df['SquareFeet'] >= SquareFeet - 20) & (df['SquareFeet'] <= SquareFeet + 20) & (df['Bedrooms'] >= Bedrooms - 0) & (df['Bedrooms'] <= Bedrooms + 0) & (df['Bathrooms'] >= Bathrooms - 0) & (df['Bathrooms'] <= Bathrooms + 0) & (df['YearBuilt'] >= YearBuilt - 5) & (df['YearBuilt'] <= YearBuilt + 5)];
    df_list = filter_df.values.tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        customdata=df_list,
        x = filter_df['SquareFeet'].tolist(),
        y = filter_df['Price'].tolist(), 
        mode = 'markers', 
        marker = dict(color = 'blue'), 
        hoverinfo="text", 
        hovertemplate=  '<b>Square Feet</b>: %{x}<br><b>Bedrooms</b>: %{customdata[1]}<br><b>Bathrooms</b>: %{customdata[2]}<br><b>Neighborhood</b>: %{customdata[3]}<br><b>Year Built</b>: %{customdata[4]}<br><b>Price</b>: %{y}<extra></extra>',
        name = 'House Price Actual'
    ))

    fig.add_trace(go.Scatter(
        x = [SquareFeet], 
        y = [YPredict], 
        mode = 'markers', 
        marker = dict(color = 'red'), 
        hovertext = 'Predicted Price',
        name = 'House Price Prediction'
    ))

    fig.update_layout()

    return YPredict, fig

with gr.Blocks() as demo:
    gr.Markdown("""
                # House Price Prediction
                This is a simple model to predict the price of a house based on its features. The database's feature have min sqaure feet is about 1000 and max is 3000.
                """)
    gr.Markdown("""
                Enter the features of the house and click 'Predict Price' to see the predicted price. 
                You can also click 'Filter Map' to see the actual prices of houses with similar features on a map.
                    """)
    with gr.Column():
        with gr.Row():
            SquareFeet = gr.Number(value=250, label="Square Feet")
            Bedrooms = gr.Number(value=3, label="Bedrooms")
            Bathrooms = gr.Number(value=1, label="Bathrooms")
            Neighborhood = gr.Radio(["Suburb", "Urban", "Rural"], label="Neighborhood")
            YearBuilt = gr.Number(value=2020, label="Year Built")
    gr.Button("Predict Price").click(
        fn=HousePrice,
        inputs=[SquareFeet, Bedrooms, Bathrooms, Neighborhood, YearBuilt],
        outputs=[gr.Textbox(label="Predicted Price"), gr.Plot(label="Similar Houses")]
    )
demo.launch()
