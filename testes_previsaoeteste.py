import numpy as np
import pandas as pd
import pandas_datareader as pdr
import datetime as dt

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.dates as dates


#sh = book.sheet_by_index(0)
#print(sh.name, sh.nrows, sh.ncols)
#print("Valor da célula D30 é ", sh.cell_value(rowx=29, colx=3))
#for rx in range(sh.nrows):
#    print(sh.row(rx))

dframe = pd.read_csv("/home/sauloramos/Documents/Trading/listadecodigos.csv", usecols=[0])
#print (df)
lt = dframe.values.tolist()

start = dt.datetime.now() - dt.timedelta(days=5*365)
end = dt.datetime.now() #- dt.timedelta(days=1)
print (end)
print (lt)

listofdf = []
 
melhorconfiancaint = 0
melhorconfiancaintarray = []
melhorconfiancastring = []


for i in lt:

    print ("---------------- ", i, " --------------" )
    #Get the stock quote
    df = pdr.DataReader(i, data_source='yahoo', start=start, end=end)
    #Show teh data

    df = df[['Close']]

    forecast_out = 1

    df['Prediction'] = df[['Close']].shift(-forecast_out)

    #print(df.tail())

    X = np.array(df.drop(['Prediction'],1))

    X = X[:-forecast_out]
    #print(X)

    y = np.array(df['Prediction'])
    
    y = y[:-forecast_out]

    #Split the data into 80% training and 20% testing

    x_train, x_test, y_train, y_test = train_test_split (X, y, test_size=0.054794521)

    # Create and train the Support Vector Machine (Regressor)

    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    svr_rbf.fit(x_train, y_train)

    svm_confidence = svr_rbf.score (x_test, y_test)


    lr = LinearRegression()

    lr.fit(x_train, y_train)

    lr_confidence = lr.score (x_test, y_test)


    x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
    lr_prediction = lr.predict(x_forecast)
    print("LR Confidence: ", lr_confidence)
    

    # Print support vector regressor model predictions for the next 'n' days
    svm_prediction = svr_rbf.predict(x_forecast)
    

    # Print linear regression model predictions for the next 'n' days

    print ("SVM Confidence", svm_confidence)
    
    print("Forecast FECHADO", x_forecast)
    print("LR PREVISTO: ", lr_prediction)
    print("SVM PREVISTO: ", svm_prediction)

    if (lr_confidence > melhorconfiancaint and lr_confidence != 1):
        melhorconfiancaint = lr_confidence
        melhorconfiancaintarray.append(lr_confidence)
        melhorconfiancastring.append(i)

    # print (df.tail())
    #listofdf.append(df)
    # plt.figure(figsize=(16,8))
    # plt.title(i)
    # plt.suptitle('Close Price History')
    # plt.plot(df['Close'])
    # plt.xlabel('Date', fontsize=18)
    # plt.ylabel('Close Price R\$ ($)', fontsize=18)
    # plt.show(block=False)
    # plt.pause(0.2)
    # plt.close()

for i in range(0,len(melhorconfiancaintarray)):
    print ("Confiaca: ",melhorconfiancaintarray[i], "--- Acao: ", melhorconfiancastring[i])

#Visualize the closing price history

# for i in range(len(listofdf)):
    
#     plt.figure(figsize=(16,8))
#     plt.title(lt[i])
#     plt.suptitle('Close Price History')
#     plt.plot(listofdf[i]['Close'])
#     plt.xlabel('Date', fontsize=18)
#     plt.ylabel('Close Price R\$ ($)', fontsize=18)
#     plt.show(block=False)
#     plt.pause(0.01)
#     plt.close()