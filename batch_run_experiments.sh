#!/bin/bash

for i in {1..5}
do
    echo "Running T$i"
    python stats_forecasting.py covid_deaths -s "T$i" -f 5
    python stats_forecasting.py covid_deaths -s "T$i" -f 10
done