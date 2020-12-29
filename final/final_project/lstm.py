# Import the libraries
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow.python.keras.layers import Dense, LSTM
from tensorflow.python.keras import Sequential
from keras import metrics
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import math
from plotly import graph_objs as go
from PIL import Image

plt.style.use('fivethirtyeight')

st.title('Amazon Stock App: LSTM')


@st.cache
def load_data():
    pd_data = pd.read_csv('./data/AMZN.csv')
    return pd_data


# Visualize the closing price history
def visualize_closing_price_history(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.Date, y=df['Adj Close'], name="Adj Close Price USD ($)", line_color='deepskyblue'))
    fig.layout.update(title_text='Adj Close Price History')
    st.plotly_chart(fig)
    return fig


def create_training_dataset_and_scaled_training_dataset(df):
    # Create a new dataframe with only the 'Close column
    st.markdown("1. Создадим новый dataframe только со столбцом 'Adj Close'.")
    dataframe = df.filter(['Adj Close'])
    st.markdown("Визуализируем историю цен закрытия ('Adj Close'):")
    visualize_closing_price_history(df)
    # Convert the dataframe to a numpy array
    dataset = dataframe.values
    # Get the number of rows to train the model on
    training_data_len = math.ceil(len(dataset) * .8)
    # training_data_len = 202

    # Scale the data
    st.markdown("2. Масштабируем данные с помощью MinMaxScaler")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Create the training data set
    # Create the scaled training data set
    st.markdown("3. Создадим training_dataset и scaled_training_dataset")
    train_data = scaled_data[0:training_data_len, :]
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # x_train = (142, 60, 1)
    return x_train, y_train, scaled_data, training_data_len, scaler, dataset, dataframe


def create_testing_dataset(scaled_data, training_data_len, model, scaler, dataset):
    # Create the testing data set
    # Create a new array containing scaled values from index 1543 to 2002
    test_data = scaled_data[training_data_len - 60:, :]
    # Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    # Convert the data to a numpy array
    x_test = np.array(x_test)
    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    # Get the models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    return predictions, y_test


data = load_data()
x_train, y_train, scaled_data, training_data_len, scaler, dataset, dataframe = create_training_dataset_and_scaled_training_dataset(
    data)

st.markdown("4. Построим LSTM модель")
st.markdown("Выберите критерии для построения:")
user_input50 = st.number_input("Lstm units", 1)
user_input25 = st.number_input("Dense units", 1)
# Build the LSTM model
st.text("model = Sequential()")
st.text("model.add(LSTM(%s, return_sequences=True, input_shape=(x_train.shape[1], 1)))" % user_input50)
st.text("model.add(LSTM(%s, return_sequences=False)))" % user_input50)
st.text("model.add(Dense(25))")
st.text("model.add(Dense(1))")
model = Sequential()
model.add(LSTM(user_input50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(user_input50, return_sequences=False))
model.add(Dense(user_input25))
model.add(Dense(1))
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
# Train the model
model.fit(x_train, y_train, epochs=1, batch_size=1)

predictions, y_test = create_testing_dataset(scaled_data, training_data_len, model, scaler, dataset)

# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
# # Visualize the data
# plt.figure(figsize=(16, 8))
# plt.title('Model')
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Adj Close Price USD ($)', fontsize=18)
# plt.plot(train['Adj Close'])
# plt.plot(valid[['Adj Close', 'Predictions']])
# plt.legend(['Train', 'Test', 'Predictions'], loc='lower right')
# plt.show()

spec = {
    "mark": "line",
    "encoding": {
        "x": {"field": train['Adj Close'], "type": "quantitative"},
        "y": {"field": valid[['Adj Close', 'Predictions']], "type": "quantitative"},
    },
}

st.subheader("View dependence on two variables")
plt_2_variables = st.vega_lite_chart(spec, width=500, height=300)
plt_2_variables.vega_lite_chart(data, spec)
#
# def visualize_closing_price_history2(df):
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=df.Date, y=train['Adj Close'], name="Train", line_color='deepskyblue'))
#     fig.add_trace(go.Scatter(x=df.Date, y=valid[['Adj Close', 'Predictions']], name=["Test", "Predictions"], line_color='dimgray'))
#     fig.layout.update(title_text='Adj Close Price History')
#     st.plotly_chart(fig)
#     return fig

# visualize_closing_price_history2(data)

image = Image.open('im.png')
st.image(image, use_column_width=True)

# Get the quote
quote_quote = load_data()
# Create a new dataframe
new_df = quote_quote.filter(['Adj Close'])
# Get teh last 60 day closing price values and convert the dataframe to an array
last_60_days = new_df[-60:].values
# Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
# Create an empty list
X_test = []
# Append teh past 60 days
X_test.append(last_60_days_scaled)
# Convert the X_test data set to a numpy array
X_test = np.array(X_test)
# Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# Get the predicted scaled price
pred_price = model.predict(X_test)
# undo the scaling
pred_price = scaler.inverse_transform(pred_price)
st.markdown("5. Предскажем 253 значение:")
st.text(pred_price)

# Calculate and print values of MAE, MSE, RMSE
st.markdown("6. Метрики")
st.text('Mean Absolute Error: %s' % metrics.mean_absolute_error(y_test, predictions))
st.text('Mean Squared Error: %s' % metrics.mean_squared_error(y_test, predictions))
st.text('Mean Absolute Percentage Error: %s' % np.mean(np.abs((predictions - y_test) / y_test) * 100))
st.text('Root Mean Squared Error: %s' % np.sqrt(np.mean(((predictions - y_test) ** 2))))
