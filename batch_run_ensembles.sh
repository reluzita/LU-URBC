#!/bin/bash

for i in {1..5}
do
    echo "Running T$i"
    python ensemble_forecasting.py covid_deaths -s "T$i"
done