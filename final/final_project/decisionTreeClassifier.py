from sklearn.metrics import accuracy_score
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


#Get the stock quote
df = pd.read_csv('AMZN.csv')
#Show teh data
df

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
            title='Price',
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='red'
        )
    )
)
df_data = [{'x':df['Date'], 'y':df['Close']}]
plot = go.Figure(data=df_data, layout=layout)

iplot(plot)

df['Open-Close']= df.Close - df.Open
df['High-Low']  = df.High - df.Low
df = df.dropna()
X= df[['Open-Close', 'High-Low']]
X.head()

Y= np.where(df['Adj Close'].shift(-1)>df['Adj Close'],1,-1)


split_percentage = 0.8
split = int(split_percentage*len(df))
X_train = X[:split]
Y_train = Y[:split]
X_test = X[split:]
Y_test = Y[split:]


model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
print(model)

accuracy_train = accuracy_score(Y_train, model.predict(X_train))
accuracy_test = accuracy_score(Y_test, model.predict(X_test))

print ('Train_data Accuracy: %.2f' %accuracy_train)
print ('Test_data Accuracy: %.2f' %accuracy_test)

model.score(X_train, Y_train)


probability = model.predict_proba(X_test)
print(probability)


predicted = model.predict(X_test)


print(metrics.confusion_matrix(Y_test, predicted))


print(metrics.classification_report(Y_test, predicted))


print(model.score(X_train,Y_train))


# ACCURACY OF TRAINING MODEL FOR AMAZON STOCK MARKET PRICE PREDICTION IS ABOUT 100%

pipe_line = Pipeline([('clf', DecisionTreeClassifier())])
pipe_line.fit(X_train, Y_train)
pipe_line.score(X_train, Y_train)

score = cross_val_score(estimator=pipe_line, X=X, y=Y, cv=10)

print('cv accuracy score : %s' % score)
print('cv accuracy : %.3f +/- %.3f' % (np.mean(score), np.std(score)))

st.text('Mean Absolute Error: %s' % metrics.mean_absolute_error(Y_test, predicted))
st.text('Mean Squared Error: %s' % metrics.mean_squared_error(Y_test,predicted ))
st.text('Mean Absolute Percentage Error: %s' % np.mean(np.abs((predicted - Y_test ) / Y_test)*100) - 100)
st.text('Root Mean Squared Error: %s' % np.sqrt(np.mean(((predicted- Y_test)**2))))