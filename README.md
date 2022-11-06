# LSTM-Networks-Applied-to-Exchange-Rate-Forecasting

Exchange rate forecasting has always been a challenge for governments, industries,
investors, and the academic community. There are so many factors that can affect market
behavior, in such a way that a mathematical description of these relationships is almost
impossible. However, artificial neural networks have reached great prediction results in the
last years, showing themselves capable of capturing the non-linearity of financial markets
relationships. This paper examines the potential of LSTM for exchange rate forecasting. A
total of 5.760 historical records were used during the training phase. Once the model has
learned the relationships from the training dataset, 1.440 new records are used to test the
model and evaluate its performance. For benchmarking purposes, standard RNN was applied
to solve the same prediction problem. __Results showed better performance of LSTM over
standard RNN, achieving error rates lower by 71.8%.__

[LSTM Networks Applied to Exchange Rate Forecasting - PDF](https://github.com/brunoRenzo6/LSTM-Networks-Applied-to-Exchange-Rate-Forecasting/blob/main/TCC_paper/TCC_BrunoNRenzo.pdf)
</br>
[LSTM Networks Applied to Exchange Rate Forecasting - JupyterNotebook](https://github.com/brunoRenzo6/LSTM-Networks-Applied-to-Exchange-Rate-Forecasting/blob/main/TCC_scripts/stacked_lstm.ipynb)

## LSTM x Vanilla RNN
<table>
  <tr>
    <td>
      <img src="https://github.com/brunoRenzo6/LSTM-Networks-Applied-to-Exchange-Rate-Forecasting/blob/main/TCC_paper/figuras/results/testPrediction.JPG"/>
      </br>
      LSTM - Prediction results on test dataset
    </td>
    <td>
      <img src="https://github.com/brunoRenzo6/LSTM-Networks-Applied-to-Exchange-Rate-Forecasting/blob/main/TCC_paper/figuras/results/testPrediction_RNN.JPG" />
      </br>
      Vanilla RNN - Prediction results on test dataset
    </td>
  </tr>
</table>

## Performance Metrics
<img src="https://github.com/brunoRenzo6/LSTM-Networks-Applied-to-Exchange-Rate-Forecasting/blob/main/TCC_paper/figuras/results/performanceMetric.PNG"/>
