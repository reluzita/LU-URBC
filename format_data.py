import pandas as pd
import datetime
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'format_data.py',
                    description = 'Format dataset for forecasting')

    parser.add_argument('dataset', type=str, help='dataset name')
    args = parser.parse_args()

    data = {}
    with open(f'data/raw/{args.dataset}.tsf', 'r', encoding='utf-8') as f:
        lines = f.readlines()

        series = lines[lines.index('@data\n')+1:]
        date = series[0].split(":")[1]
        for ts in series:
            seq = ts.split(":")
            data[seq[0]] = [float(x) for x in seq[2].strip().split(',')]

    data['date'] = [datetime.datetime.strptime(date, '%Y-%m-%d %H-%M-%S') + datetime.timedelta(days=i) 
        for i in range(len(data['T1']))]

    data = pd.DataFrame(data).set_index('date')
    data.to_csv(f'data/clean/{args.dataset}.csv')