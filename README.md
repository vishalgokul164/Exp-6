# EXPERIMENT 6 : ARIMA MODEL FOR FORECASTING

## AIM

To evaluate predictive models using performance metrics.

## ALGORITHM

1.  *START*
2.  Prepare stationary time series.
3.  Identify ARIMA -- (p, d, q) parameters using ACF/PACF.
4.  Fit the ARIMA model on the data.
5.  Forecast future periods and plot the results.
6.  Evaluate forecast accuracy.
7.  *STOP*



## PROGRAM

import matplotlib.pyplot as plt  
import numpy as np  
import pandas as pd  
from statsmodels.tsa.arima.model import ARIMA  
from statsmodels.tsa.stattools import adfuller  
print("--- Time Series Data Preparation ---")  
np.random.seed(42)  
date_rng = pd.date_range(start="2024-01-01", periods=120, freq="ME")  
trend = np.linspace(10, 50, 120)    
noise = np.random.normal(0, 2, 120)  
ts_data = trend + noise  
print(df.head())  
adf_result = adfuller(df["Value"])  
print(f"\nADF Statistic: {adf_result[0]:.4f}")  
print(f"p-value: {adf_result[1]:.4f}")  
print("Time series rendered stationary (or differenced appropriately).")  
train_data = df.iloc[:-12]  
test_data = df.iloc[-12:]  
print("\n--- ARIMA Model Training ---")              
model = ARIMA(train_data["Value"], order=(1, 1, 1))  
fitted_model = model.fit()    
print(fitted_model.summary())    
print("\n--- Forecasting ---")  
forecast_result = fitted_model.get_forecast(steps=12)  
forecast_mean = forecast_result.predicted_mean  
forecast_ci = forecast_result.conf_int()  
print("\n--- Model Evaluation ---")  
actuals = test_data["Value"].values  
predictions = forecast_mean.values  
mae = np.mean(np.abs(actuals - predictions))  
rmse = np.sqrt(np.mean((actuals - predictions) ** 2))    
print(f"Mean Absolute Error (MAE): {mae:.4f}")    
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")  
print("\n--- Generating Forecast Plot ---")  
plt.figure(figsize=(12, 6))  
plt.plot(train_data.index, train_data["Value"], label="Training Data", color="blue")  
plt.plot(test_data.index, test_data["Value"], label="Actual Test Data", color="green")  
plt.plot(  
    forecast_mean.index,    
    forecast_mean,  
    label="ARIMA Forecast",  
    color="red",    
    linestyle="--",  
)  
plt.fill_between(        
    forecast_ci.index,        
    forecast_ci.iloc[:, 0],    
    forecast_ci.iloc[:, 1],    
    color="pink",  
    alpha=0.3,  
    label="Confidence Interval",  
)  

plt.title("ARIMA Model Time Series Forecasting")  
plt.xlabel("Date")  
plt.ylabel("Value")  
plt.legend()  
plt.grid(True, linestyle="--", alpha=0.6)  
plt.tight_layout()  
print("Forecast plot generated and rendered successfully.")  
plt.show()    
  
## OUTPUT

<img width="707" height="550" alt="image" src="https://github.com/user-attachments/assets/a6d0555f-6da8-45f9-b5b5-4ceaccc07751" />
<img width="620" height="142" alt="image" src="https://github.com/user-attachments/assets/9909d467-5713-4048-ae56-8b55bfcf06b8" />
<img width="988" height="477" alt="image" src="https://github.com/user-attachments/assets/a3f8a91d-1c67-4cc9-a1d3-79733d5a6a25" />


## RESULT   

The ARIMA $(1, 1, 1)$ time series forecasting model was successfully implemented using Python's statsmodels library. After verifying stationarity, the model was fitted to the historical trend data to predict subsequent periods. Performance evaluation using *MAE* and *RMSE* demonstrated that the ARIMA model effectively captures temporal dependencies and generates reliable future 
