import argparse
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from math import sqrt
from itertools import product
from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError

from tbats import TBATS
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox


def optimize_ARIMA(order_list, exog):
    """
        Return dataframe with parameters and corresponding AIC
        
        order_list - list with (p, d, q) tuples
        exog - the exogenous variable
    """
    
    results = []
    
    for order in tqdm(order_list):
        try: 
            model = SARIMAX(exog, order=order).fit(disp=-1)
        except:
            continue
            
        aic = model.aic
        results.append([order, model.aic])
        
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p, d, q)', 'AIC']
    #Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return result_df

if __name__=='__main__': 
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(
                    prog = 'stats_forecasting.py',
                    description = 'Forecasting using statistical models')

    parser.add_argument('dataset', type=str, help='dataset name')
    parser.add_argument('-s', '--seriesname', type=str, help='column name of the series')
    parser.add_argument('-f', '--horizon', type=int, help='forecasting horizon')

    args = parser.parse_args()

    data = pd.read_csv(f'data/clean/{args.dataset}.csv').set_index('date')
    X = data[args.seriesname].values
    train, test = X[1:-args.horizon], X[-args.horizon:]

    predictions = {}
    errors = {}

    # dummy prediction
    print('Dummy prediction...')
    predictions['dummy'] = []
    for x in range(args.horizon):
        predictions['dummy'].append(train[-1])

    # autoregressive model
    print('Fitting autoregressive model...')
    model = AutoReg(train, lags=10).fit() # otimizar lags??
    predictions['autoregression'] = model.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

    # moving average model
    print('Fitting moving average model...')
    predictions['moving average'] = data[args.seriesname].rolling(window=10).mean()[-args.horizon:].values

    # ARMA model
    print('Fitting ARMA model...')
    ps = range(0, 8, 1)
    d = 0
    qs = range(0, 8, 1)
    # Create a list with all possible combination of parameters
    parameters = product(ps, qs)
    parameters_list = list(parameters)
    order_list = []
    for each in parameters_list:
        each = list(each)
        each.insert(1, d)
        each = tuple(each)
        order_list.append(each)
        
    result_df = optimize_ARIMA(order_list, exog=train)
    best_order = result_df['(p, d, q)'][0]
    model = SARIMAX(train, order=best_order).fit(disp=-1)
    predictions['ARMA'] = model.forecast(args.horizon)

    # TBATS
    print('Fitting TBATS model...')
    model = TBATS().fit(train)
    predictions['TBATS'] = model.forecast(steps=args.horizon)

    # calculate errors
    mase = MeanAbsoluteScaledError()
    for modelname in predictions:
        errors[modelname] = mase(test, np.array(predictions[modelname]), y_train=train)

    errors_df = pd.DataFrame.from_dict(errors, orient='index', columns=['MASE'])
    errors_df.to_csv(f'results/errors/{args.dataset}_{args.seriesname}.csv')

    # save predictions
    predictions_df = pd.DataFrame.from_dict(predictions, orient='index', columns=list(data.index)[-args.horizon:])
    predictions_df.to_csv(f'results/predictions/{args.dataset}_{args.seriesname}.csv')

  
