import pandas as pd
import argparse
from sklearn.linear_model import ElasticNet


if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'ensemble_forecasting.py',
                    description = 'Forecasting using stacking ensemble')

    parser.add_argument('dataset', type=str, help='dataset name')
    parser.add_argument('-s', '--seriesname', type=str, help='column name of the series')

    args = parser.parse_args()

    predictions = pd.read_csv(f'results/predictions/{args.dataset}_{args.seriesname}_10.csv', index_col=0)
    data = pd.read_csv(f'data/clean/{args.dataset}.csv', index_col=0)

    X = pd.DataFrame(columns=predictions.index)
    for date in predictions.columns:
        X.loc[date] = predictions[date].values

    X_train = X[:5].values
    X_test = X[5:].values

    y_train = data[args.seriesname][-10:-5]
    y_test = data[args.seriesname][-5:]

    regr = ElasticNet(random_state=0).fit(X_train, y_train)
    y_pred = regr.predict(X_test)

    predictions_5 = pd.read_csv(f'results/predictions/{args.dataset}_{args.seriesname}_5.csv', index_col=0)
    predictions_5.loc['ensemble'] = y_pred
    predictions_5.to_csv(f'results/predictions/{args.dataset}_{args.seriesname}_5.csv')