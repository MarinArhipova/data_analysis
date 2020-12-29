import streamlit as st
# Import package for linear model
from sklearn.linear_model import LinearRegression
# Import package for splitting data set
from sklearn.model_selection import train_test_split
# Import metrics package from sklearn for statistical analysis
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Регрессия ищет отношения между переменными, то есть вам нужно найти функцию,
# которая отображает зависимость одних переменных или данных от других.

# В данном случае рассматривается зависимость price от time


data = pd.read_csv('./data/AMZN.csv')
df = pd.DataFrame(data, columns=['Adj Close'])
df = df.reset_index()


# Split data into train and test set: 80% / 20%
train, test = train_test_split(df, test_size=0.20)
# Reshape index column to 2D array for .fit() method
X_train = np.array(train.index).reshape(-1, 1)
y_train = train['Adj Close']

# Create LinearRegression Object
model = LinearRegression()
# Fit linear model using the train data set
model.fit(X_train, y_train)
model = LinearRegression()
# Fit linear model using the train data set
model.fit(X_train, y_train)

# Train set graph
plt.figure(1, figsize=(16,10))
plt.title('Linear Regression | Price vs Time')
plt.scatter(X_train, y_train, edgecolor='w', label='Actual Price')
plt.plot(X_train, model.predict(X_train), color='r', label='Predicted Price')
plt.xlabel('Integer Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

X_test = np.array(test.index).reshape(-1, 1)
y_test = test['Adj Close']

# Generate array with predicted values
y_pred = model.predict(X_test)
df['Prediction'] = model.predict(np.array(df.index).reshape(-1, 1))

df.head()

# Generate 25 random numbers
randints = np.random.randint(252, size=25)
# Select row numbers == random numbers
df_sample = df[df.index.isin(randints)]

df_sample.head()

# Plot fitted line, y test
# plt.figure(1, figsize=(16,10))
# plt.title('Linear Regression | Price vs Time')
# plt.plot(X_test, model.predict(X_test), color='r', label='Predicted Price')
# plt.scatter(X_test, y_test, edgecolor='w', label='Actual Price')
# plt.xlabel('Integer Date')
# plt.ylabel('Stock Price in $')
# plt.show()

spec = {
    "mark": "line",
    "encoding": {
        "x": {"field": X_test, "type": "quantitative"},
        "y": {"field": y_test, "type": "quantitative"},
    },
}

st.subheader("View dependence on two variables")
plt_2_variables = st.vega_lite_chart(spec, width=500, height=300)
plt_2_variables.vega_lite_chart(data, spec)

# Calculate and print values of MAE, MSE, RMSE
st.markdown("Метрики")
st.text('Mean Absolute Error: %s' % metrics.mean_absolute_error(y_test, y_pred))
st.text('Mean Squared Error: %s' % metrics.mean_squared_error(y_test, y_pred))
st.text('Mean Absolute Percentage Error: %s' % np.mean(np.abs((y_test - y_pred) / y_test)*100))
st.text('Root Mean Squared Error: %s' % np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
st.text('R2: %s' % metrics.r2_score(y_test, y_pred))