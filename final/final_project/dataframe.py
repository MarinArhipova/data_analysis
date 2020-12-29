import streamlit as st
import pandas as pd
from plotly import graph_objs as go
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
# LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px
# LSTM
import math
from tensorflow.python.keras.layers import Dense, LSTM
from tensorflow.python.keras import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from PIL import Image
# DTC
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


st.title('Amazon Stock App')
st.markdown("Основная цель этого исследования - проанализировать и спрогнозировать курс акций компании Amazon.")
DATA_URL = ('./data/AMZN.csv')


@st.cache(allow_output_mutation=True)
def load_data():
    pd_data = pd.read_csv(DATA_URL)
    return pd_data


data = load_data()
data.shape
#(252, 7)

# volume - Количество проданных акций
# Adjusted Close = adj close - Скорректированное закрытие
st.text("Open - цена акции в начале торгового дня (это не обязательно должна быть цена закрытия предыдущего торгового дня)")
st.text("High - самая высокая цена акции в этот торговый день")
st.text("Low - самая низкая цена акции в этот торговый день")
st.text("Close - цена акции на момент закрытия торгового дня ")
st.text("Adj close - Adjusted Close - скорректированное закрытие - цена акции, с учетом цен на корпоративные действия. Хотя считается, что цены на акции устанавливаются в основном трейдерами, дробление акций и дивиденды (выплата прибыли компании на акцию) также влияют на цену акции и должны учитываться.")
st.text("Volume - количество проданных акций")


def plot_fig():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.Date, y=data['Open'], name="stock_open", line_color='deepskyblue'))
    fig.add_trace(go.Scatter(x=data.Date, y=data['Close'], name="stock_close", line_color='dimgray'))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    return fig
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)


st.markdown("Построим график, чтобы мы могли визуально видеть изменение цены. Как вы могли заметить, акции сильно выросли из-за пандемии.")
plot_fig()


st.markdown("Далее мы создадим новый DataFrame только с ценой и датой закрытия и продолжим работу с ним.")
df = pd.DataFrame(data, columns=['Adj Close'])
# Reset index column so that we have integers to represent time for later analysis
# Сбросим столбец индекса, чтобы у нас были целые числа для представления времени для последующего анализа
df = df.reset_index()
# Проверим наличие пропущенных значений в столбцах
df.isna().values.any()
# False


st.markdown("Выберите одну из моделей для дальнейшего исследования.")
select = st.selectbox('Select model', ['None', 'Linear Regression', 'LSTM', 'Decision tree classifier'], key='1')
st.write('You selected:', select)

#########################################
#            Linear Regression          #
#########################################
def alg_lr():
    # Split data into train and test set: 80% / 20%
    train, test = train_test_split(df, test_size=0.20)
    # Reshape index column to 2D array for .fit() method
    X_train = np.array(train.index).reshape(-1, 1)
    y_train = train['Adj Close']

    # Create LinearRegression Object
    st.write("Предскажем значения:")
    st.write("model = LinearRegression()")
    model = LinearRegression()
    # Fit linear model using the train data set
    model.fit(X_train, y_train)
    model = LinearRegression()
    # Fit linear model using the train data set
    model.fit(X_train, y_train)

    # plt.figure(1, figsize=(16, 10))
    # plt.title('Linear Regression | Price vs Time')
    # plt.scatter(X_train, y_train, edgecolor='w', label='Actual Price')
    # plt.plot(X_train, model.predict(X_train), color='r', label='Predicted Price')
    # plt.xlabel('Integer Date')
    # plt.ylabel('Stock Price')
    # plt.legend()
    # plt.show()

    # fig = px.scatter(x=X_train, y=y_train, title="Linear Regression | Price vs Time")
    # fig.add_trace(go.Scatter(x=X_train, y=y_train, name="Adj Close Price USD ($)", line_color='deepskyblue'))
    # fig.add_trace(go.Scatter(x=X_train, y=model.predict(X_train), name="Predicted Price", line_color='dimgray'))
    # fig.layout.update(title_text='Linear Regression | Price vs Time')
    # st.plotly_chart(fig)
    image = Image.open('lr.png')
    st.image(image, use_column_width=True)

    X_test = np.array(test.index).reshape(-1, 1)
    y_test = test['Adj Close']

    # Generate array with predicted values
    y_pred = model.predict(X_test)
    df['Prediction'] = model.predict(np.array(df.index).reshape(-1, 1))
    st.write("Сгенерируем массив с предсказанными значениями")
    st.write(df.head())

    st.write("Сгенерируем 25 рандомных значений и добавим в тестовую выборку")
    randints = np.random.randint(252, size=25)
    # Select row numbers == random numbers
    df_sample = df[df.index.isin(randints)]

    # Plot fitted line, y test
    # plt.figure(1, figsize=(16, 10))
    # plt.title('Linear Regression | Price vs Time')
    # plt.plot(X_test, model.predict(X_test), color='r', label='Predicted Price')
    # plt.scatter(X_test, y_test, edgecolor='w', label='Actual Price')
    # plt.xlabel('Integer Date')
    # plt.ylabel('Stock Price in $')
    # plt.show()

    image = Image.open('lr2.png')
    st.image(image, use_column_width=True)

    # Calculate and print values of MAE, MSE, RMSE
    st.markdown("Метрики")
    st.text('Mean Absolute Error: %s' % metrics.mean_absolute_error(y_test, y_pred))
    st.text('Mean Squared Error: %s' % metrics.mean_squared_error(y_test, y_pred))
    st.text('Mean Absolute Percentage Error: %s' % np.mean(np.abs((y_test - y_pred) / y_test) * 100))
    st.text('Root Mean Squared Error: %s' % np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


#########################################
#                  LSTM                 #
#########################################
def alg_lstm(data):
    # Visualize the closing price history
    def visualize_closing_price_history(df):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=df.Date, y=df['Adj Close'], name="Adj Close Price USD ($)", line_color='deepskyblue'))
        fig.layout.update(title_text='Adj Close Price History')
        st.plotly_chart(fig)
        return fig


    def create_training_dataset_and_scaled_training_dataset(df):
        # Create a new dataframe with only the 'Close column
        st.markdown("1. Создадим новый dataframe только со столбцом 'Adj Close'")
        dataframe = df.filter(['Adj Close'])
        st.markdown("Визуализируем историю цен закрытия ('Adj Close'):")
        visualize_closing_price_history(df)
        # Convert the dataframe to a numpy array
        dataset = dataframe.values
        # Get the number of rows to train the model on
        training_data_len = math.ceil(len(dataset) * .8)
        # training_data_len = 202

        # Scale the data
        st.markdown("2. Нормализация данных с помощью MinMaxScaler")
        # LSTM чувствительны к масштабу входных данных.  Рекомендуется изменить масштаб данных в диапазоне от 0 до 1,
        # что также называется нормализацией. Мы можем легко нормализовать набор данных с помощью класса предварительной
        # обработки MinMaxScaler из библиотеки scikit-learn.
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        # Create the training data set
        # Create the scaled training data set
        st.markdown("3. Разделим набор данных на обучающий и тестовый наборы данных")
        # Для данных временных рядов важна последовательность значений. Простой метод, который мы можем использовать,
        # - разделить упорядоченный набор данных на обучающие и тестовые наборы данных.
        # Приведенный ниже разделяет данные на обучающие наборы данных с n-60 наблюдений,
        # которые мы можем использовать для обучения нашей модели, оставляя оставшиеся 60 для тестирования модели.
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

    x_train, y_train, scaled_data, training_data_len, scaler, dataset, dataframe = create_training_dataset_and_scaled_training_dataset(
        data)

    st.markdown("4. Построим LSTM модель")
    st.markdown("Выберите критерии для построения:")
    # Build the LSTM model
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


#########################################
#       Decision Tree Classifier        #
#########################################
def alg_dtc(data):
    layout = go.Layout(
        title='Amazon Stock Prices',
        xaxis=dict(
            title='date',
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='blue'
            )
        ),
        yaxis=dict(
            title='Adj Close',
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='red'
            )
        )
    )
    df_data = [{'x': data['Date'], 'y': data['Adj Close']}]
    plot = go.Figure(data=df_data, layout=layout)
    st.plotly_chart(plot)
    data['Open-Close'] = data.Close - data.Open
    data['High-Low'] = data.High - data.Low
    data = data.dropna()
    X = data[['Open-Close', 'High-Low']]
    X.head()
    Y = np.where(df['Adj Close'].shift(-1) > df['Adj Close'], 1, -1)
    split_percentage = 0.8
    split = int(split_percentage * len(df))
    X_train = X[:split]
    Y_train = Y[:split]
    X_test = X[split:]
    Y_test = Y[split:]

    model = DecisionTreeClassifier()
    st.text('model = DecisionTreeClassifier()')
    model.fit(X_train, Y_train)
    accuracy_train = accuracy_score(Y_train, model.predict(X_train))
    accuracy_test = accuracy_score(Y_test, model.predict(X_test))

    st.text('Train_data Accuracy: %.2f' % accuracy_train)
    st.text('Test_data Accuracy: %.2f' % accuracy_test)

    model.score(X_train, Y_train)

    probability = model.predict_proba(X_test)

    predicted = model.predict(X_test)

    # print(metrics.confusion_matrix(Y_test, predicted))
    # print(metrics.classification_report(Y_test, predicted))
    # st.text(model.score(X_train, Y_train))

    st.text('ACCURACY OF TRAINING MODEL FOR AMAZON STOCK MARKET PRICE PREDICTION IS ABOUT 100%')
    st.text('')

    pipe_line = Pipeline([('clf', DecisionTreeClassifier())])
    pipe_line.fit(X_train, Y_train)
    pipe_line.score(X_train, Y_train)

    score = cross_val_score(estimator=pipe_line, X=X, y=Y, cv=10)
    # print('cv accuracy score : %s' % score)
    # print('cv accuracy : %.3f +/- %.3f' % (np.mean(score), np.std(score)))

    st.markdown("Метрики")
    st.text('Mean Absolute Error: %s' % metrics.mean_absolute_error(Y_test, predicted))
    st.text('Mean Squared Error: %s' % metrics.mean_squared_error(Y_test, predicted))
    st.text('Root Mean Squared Error: %s' % np.sqrt(np.mean(((predicted - Y_test) ** 2))))


if select == 'Linear Regression':
    alg_lr()
if select == 'LSTM':
    alg_lstm(data)
if select == 'Decision tree classifier':
    alg_dtc(data)