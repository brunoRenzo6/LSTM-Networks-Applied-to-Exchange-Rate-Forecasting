import pandas as pd
import matplotlib.pyplot as plt
import numpy
import numpy as np
from numpy import array


def getTrainPredictPlot(df, time_step, train_data, train_predict):
    look_back=time_step
    
    trainPredictPlot = numpy.empty_like(df)
    trainPredictPlot[:,:]=np.nan

    # shift train predictions for plotting
    initialx_train=look_back
    finalx_train=initialx_train+len(train_predict)

    trainPredictPlot[initialx_train:finalx_train,:]=train_predict
    
    return trainPredictPlot

def getTestPredictPlot(df, time_step, train_data, test_predict):
    look_back=time_step
    
    testPredictPlot = numpy.empty_like(df)
    testPredictPlot[:,:]=numpy.nan

    # shift test predictions for plotting 
    initialx_test=len(train_data)+(look_back) 
    finalx_test=initialx_test+len(test_predict)

    testPredictPlot[initialx_test:finalx_test,:]=test_predict
    
    return testPredictPlot

def getFullPrediction_graph(df, time_step, train_data, train_predict, test_predict):
    fig = plt.figure(figsize = (18,9))
    #plt.xlim(6500,6560)
    #plt.ylim(3.5, 6)
    
    fig.suptitle('Train & Test Prediction', fontsize=20)
    plt.subplots_adjust(wspace=2, hspace=10)

    trainPredictPlot = getTrainPredictPlot(df, time_step, train_data, train_predict)
    testPredictPlot = getTestPredictPlot(df, time_step, train_data, test_predict)

    # plot baseline and predictions 
    plt.plot(df.values)
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.legend(['Real data','Train prediction','Test prediction'], loc='upper left')
    
    return fig

def getTestPrediction_graph(y_test, test_predict):
    fig = plt.figure(figsize = (18,9))
    #plt.xlim(6500,6560)
    #plt.ylim(3.5, 6)
    
    fig.suptitle('Test Prediction', fontsize=20)
    plt.subplots_adjust(wspace=2, hspace=10)
    
    # plot baseline and predictions 
    plt.plot(y_test)
    plt.plot(test_predict,color='green')
    plt.legend(['Real data','Train prediction','Test prediction'], loc='upper left')

    plt.show()

    return fig

def applyZoom(ax, test_data_unf, predict_position, n_predictSteps, zoom_configs): 
    xAxix_zoomOut = zoom_configs["xAxix_zoomOut"]
    yAxix_zoomOut = zoom_configs["yAxix_zoomOut"]
    
    #ZOOM
    xa,xb=(predict_position,predict_position+n_predictSteps)
    ya,yb=(min(test_data_unf[xa:xb].values),max(test_data_unf[xa:xb].values))
    
    ax.set_xlim(xa*(1-xAxix_zoomOut),xb*(1+xAxix_zoomOut))
    ax.set_ylim(ya*(1-yAxix_zoomOut),yb*(1+yAxix_zoomOut))
    
    return ax

def getTestPredictionZoomed_graph(y_test,
                                  test_data_unf,
                                  test_predict,
                                  predict_position,
                                  n_predictSteps,
                                  zoom_configs):

    # plot baseline and predictions 
    fig, ax = plt.subplots(figsize = (18,9))
    fig.suptitle('Test Prediction - Zoom', fontsize=20)
    plt.subplots_adjust(wspace=20, hspace=10)
    
    ax = applyZoom(ax, test_data_unf, predict_position, n_predictSteps, zoom_configs)
    
    #plt.xticks(range(int(xlimMin), int(xlimMax)))
    
    plt.plot(y_test)
    plt.plot(test_predict,color='green')
    plt.legend(['Real data','Train prediction','Test prediction'], loc='upper left')
    
    return fig


#####################################################################################

def lstmPredict_InSequence(model, scaler, time_step, test_data, predict_position, n_predictSteps):
    #print("len(test_data): "+str(len(test_data)))
    
    # create x_input from test_data reshaped
    x_input=test_data[predict_position-time_step:predict_position].reshape(1,-1)
    print("x_input.shape: "+str(x_input.shape))
    
    # create temp_input from x_input
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    
    lst_output=[]
    model_timeSteps=100
    i=0
    while(i<n_predictSteps):

        if(len(temp_input)>time_step):
            x_input=np.array(temp_input[1:])
            x_input=x_input.reshape(1,-1)
            
            x_input = x_input.reshape((1, model_timeSteps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            lst_output.extend(yhat.tolist())
            
            i=i+1
        else:
            x_input = x_input.reshape((1, model_timeSteps,1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            
            i=i+1
            
    return lst_output


def predictLSTM(df, model, scaler, time_step, test_data_unf, test_data,
                predict_position, n_predictSteps, zoom_configs):
    
    #LSTM PREDICT IN SEQUENCE()
    lst_output = lstmPredict_InSequence(model, scaler, time_step, test_data, predict_position, n_predictSteps)
    #print("len(lst_output):"+str(len(lst_output)))

    #APPLY ZOOM
    fig, ax = plt.subplots(figsize=(18, 9))
    ax = applyZoom(ax, test_data_unf, predict_position, n_predictSteps, zoom_configs)
    
    
    day_new=np.arange(1,len(test_data))
    day_pred=np.arange((predict_position+1),(predict_position+1+n_predictSteps))
    #print("len(df):"+str(len(df)))
    ax.plot(day_new,df[len(df)-len(day_new):len(df)],color='royalblue')
    ax.plot(day_pred,scaler.inverse_transform(lst_output),color='green')
    ax.legend(['Real data','Test prediction'], loc='upper left')
    