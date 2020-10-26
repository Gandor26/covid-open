from typing import Optional, List
from pathlib import Path
from datetime import datetime
import json
import os

import pandas as pd
import numpy as np
from . import state2abbr, abbr2state



def load_cdc_truth(
    death: bool = False,
    cumulative: bool = True,
    start_date: str = '2020-01-23',
    end_date: Optional[str] = None,
):
    url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series" 
    path = f"{url}/time_series_covid19_{'deaths' if death else 'confirmed'}_US.csv"
    
    df = pd.read_csv(path)
    data = {}
    for state in state2abbr:
        tmp = df[df['Province_State']==state].loc[:, df.columns[(12 if death else 11):]].sum(axis=0)
        tmp.index = pd.to_datetime(tmp.index)
        data[state] = tmp
    data = pd.DataFrame(data)
    if not cumulative:
        data = data.diff(1).iloc[1:]
    if end_date is not None:
        end_date = pd.to_datetime(end_date) - pd.Timedelta(1, unit='d')
    data = data.loc[start_date:end_date]
    return data

    
def load_case_baselines(
    date: str,
    est: str = 'point',
):
    time_field = 'target_end_date'
    date = f"{pd.to_datetime(date):%Y-%m-%d}"
    baselines = pd.read_csv(
        f"https://www.cdc.gov/coronavirus/2019-ncov/covid-data/files/{date}-all-forecasted-cases-model-data.csv",
        parse_dates=[time_field],
    )
    cdc = {}
    for model, data in baselines.groupby('model'):
        data = data.loc[data.fips.str.isnumeric()]
        # filter national-only
        if data.shape[0] == 0:
            continue
        if min(data.fips.str.len()) > 2:
            # aggreate county-only
            data = data.groupby(['State', time_field]).sum().reset_index()
            data['location_name'] = data['State'].apply(lambda x: abbr2state.get(x, x))
        else:
            # take state-level
            data = data[data.fips.str.len() <= 2]
        dfs = []
        for state, df in data.groupby('location_name'):
            df = df.loc[:, ['target_end_date', est]].set_index('target_end_date')
            df.index.name = 'date'
            df.columns = [state]
            dfs.append(df)
        data = pd.concat(dfs, axis=1)
        cdc[model] = data
    return cdc

def load_hosp_baselines(
    date: str,
    est: str = 'point',
):
    time_field = 'target_end_date'
    date = f"{pd.to_datetime(date):%Y-%m-%d}"
    baselines = pd.read_csv(
        f'https://www.cdc.gov/coronavirus/2019-ncov/downloads/cases-updates/{date}-hospitalizations-model-data.csv',
        parse_dates = [time_field],
    )
    cdc = {}
    for model, data in baselines.groupby('model'):
        dfs = []
        for state, df in data.groupby('location_name'):
            df = df.loc[:, [time_field, est]].set_index(time_field)
            df.index.name = 'date'
            df.columns = [state]
            dfs.append(df)
        data = pd.concat(dfs, axis=1)
        cdc[model] = data
    return cdc

    
def load_death_baselines(
    date: str,
    est: str = 'point',
):
    time_field = 'target_week_end_date'
    date = f"{pd.to_datetime(date):%Y-%m-%d}"
    baselines = pd.read_csv(
        f'https://www.cdc.gov/coronavirus/2019-ncov/covid-data/files/{date}-model-data.csv',
        parse_dates = [time_field],
    )
    cdc = {}
    for model, data in baselines.groupby('model'):
        dfs = []
        for state, df in data.groupby('location_name'):
            df = df.loc[:, [time_field, est]].set_index(time_field)
            df.index.name = 'date'
            df.columns = [state]
            dfs.append(df)
        data = pd.concat(dfs, axis=1)
        cdc[model] = data
    return cdc