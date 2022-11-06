import pandas as pd
import matplotlib.pyplot as plt

import math
from sklearn.metrics import mean_squared_error

from matplotlib.backends.backend_pdf import PdfPages




##################################################################
#
#                      Learning Curve
#
###################################################################

def applyLearningCurveConfig(axs, history, scope="model"):
    
    if scope=="model":
        axs.plot(history.history['loss'],color='royalblue')
        axs.plot(history.history['val_loss'],color='orange')
        axs.legend(['train_loss', 'validation_loss'], loc='upper left')
        axs.set_title('model loss', fontweight="bold")
        
        axs.set_ylabel('loss')
        axs.set_xlabel('epoch')
    
    elif scope=="validation":
        axs.plot(history.history['val_loss'],color='orange')
        axs.legend(['validation_loss'], loc='upper left')
        axs.set_title('validation loss', fontweight="bold")
        
    elif scope=="train":
        axs.plot(history.history['loss'],color='royalblue')
        axs.legend(['train_loss'], loc='upper left')
        axs.set_title('train loss', fontweight="bold")

    
    
    return axs

def getLearningCurve_figure(history):
    fig = plt.figure()
    fig.set_size_inches(18, 9)
    fig.suptitle('Learning Curve', fontsize=20)
    
    plt.subplots_adjust(wspace=2, hspace=10)
    
    ax1 = fig.add_subplot(10,10,(1,97))
    ax1 = applyLearningCurveConfig(ax1, history, scope="model")
    
    ax2 = fig.add_subplot(10,10,(8,50))
    ax2 = applyLearningCurveConfig(ax2, history, scope="validation")
    
    ax3 = fig.add_subplot(10,10,(58,100)) 
    ax3 = applyLearningCurveConfig(ax3, history, scope="train")
    
    plt.show()
    
    return fig



##################################################################
#
#                      MSE & RMSE
#
###################################################################
def getPerformanceMetrics_table(**kwargs):
    train_predict = kwargs.get('train_predict')
    test_predict = kwargs.get('test_predict')
    y_train = kwargs.get('y_train')
    y_test = kwargs.get('y_test')
    
    lst=[]

    trainMSE = (mean_squared_error(train_predict,y_train))
    testMSE = (mean_squared_error(test_predict,y_test))
    mse = ["MeanSquareError (MSE)",trainMSE, testMSE]
    lst.append(mse)

    trainRMSE = math.sqrt(mean_squared_error(train_predict,y_train))
    testRMSE = math.sqrt(mean_squared_error(test_predict,y_test))
    rmse = ["RootMeanSquareError (RMSE)",trainRMSE, testRMSE]
    lst.append(rmse)

    table = pd.DataFrame(lst, columns =['measure','Train', 'Test'])
    #df = df.set_index('measure')
    
    return table

def getPerformanceMetrics_figure(table):
    fig, ax =plt.subplots()
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=table.values,colLabels=table.columns,loc='center',  cellLoc="left")
    fig.tight_layout()
    
    return fig

##################################################################
#
#                      Write PDF
#
###################################################################


def write_pdf(fname, figures):
    doc = PdfPages(fname)
    for fig in figures:
        fig.savefig(doc, format='pdf')
    doc.close()
    

    
    