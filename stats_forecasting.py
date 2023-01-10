import argparse
import warnings
from itertools import product

import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tbats import TBATS
from tqdm import tqdm


def optimize_ARIMA(order_list, exog):
    results = []

    for order in tqdm(order_list):
        try:
            model = SARIMAX(exog, order=order).fit(disp=-1)
        except Exception:
            continue

        results.append([order, model, model.mse])

    best_model = min(results, key=lambda x: x[2])
    return best_model[1]


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(
        prog="stats_forecasting.py", description="Forecasting using statistical models"
    )

    parser.add_argument("dataset", type=str, help="dataset name")
    parser.add_argument(
        "-s", "--seriesname", type=str, help="column name of the series"
    )
    parser.add_argument("-f", "--horizon", type=int, help="forecasting horizon")

    args = parser.parse_args()

    data = pd.read_csv(f"data/clean/{args.dataset}.csv").set_index("date")
    X = data[args.seriesname].values
    train, test = X[1 : -args.horizon], X[-args.horizon :]

    predictions = {}
    errors = {}

    # autoregressive model
    print("Fitting autoregressive model...")
    model = AutoReg(train, lags=10).fit()  # otimizar lags??
    predictions["autoregression"] = model.predict(
        start=len(train), end=len(train) + len(test) - 1, dynamic=False
    )

    # moving average model
    print("Fitting moving average model...")
    predictions["moving average"] = (
        data[args.seriesname].rolling(window=10).mean()[-args.horizon :].values
    )

    # ARMA model
    print("Fitting ARMA model...")
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

    model = optimize_ARIMA(order_list, exog=train)
    predictions["ARMA"] = model.forecast(args.horizon)

    # TBATS
    print("Fitting TBATS model...")
    model = TBATS().fit(train)
    predictions["TBATS"] = model.forecast(steps=args.horizon)

    # save predictions
    predictions_df = pd.DataFrame.from_dict(
        predictions, orient="index", columns=list(data.index)[-args.horizon :]
    )
    predictions_df.to_csv(
        f"results/predictions/{args.dataset}_{args.seriesname}_{args.horizon}.csv"
    )
