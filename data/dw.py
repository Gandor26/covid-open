from typing import Optional, List
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import wget
import json
import os

import datadotworld as dw
import pandas as pd
import numpy as np
from .constants import state2abbr, abbr2state, pkg_root


with open(pkg_root.joinpath('token.json'), 'r') as f:
    token = json.load(f)
    os.environ['DW_AUTH_TOKEN'] = token

hosp_states = [
    'AZ', 'AR', 'CO', 'FL', 'GA', 'HI', 'ID', 'KS', 'KY', 'ME',
    'MD', 'MA', 'MN', 'MS', 'MT', 'NH', 'NM', 'NY', 'NE', 'ND', 
    'OH', 'OK', 'OR', 'RI', 'SC', 'SD', 'TN', 'UT', 'VA', 'WA', 
    'WI', 'WY',
]

ref_countries = [
    'Brazil', 
    'United Kingdom', 
    'India', 
    'Peru', 
    'Spain', 
    'Italy', 
    'Mexico',
    'Turkey', 
    'Germany', 
    'France', 
    'Colombia'
]


def load_us_covid_dataset(
    county_level: bool = False,
    death: bool = False,
    cumulative: bool = True,
    start_date: str = '2020-01-23',
    end_date: Optional[str] = None,
    selected_counties: Optional[List[str]] = None,
):
    epi_df = dw.load_dataset(
        dataset_key='covid-19-data-resource-hub/covid-19-case-counts',
        force_update=False,
        auto_update=True, 
    ).dataframes['covid_19_activity']
    ctry_col = 'country_short_name'
    state_col = 'province_state_name'
    county_col = 'county_name'
    date_col = 'report_date'
    case_col = f'''people_{f"death{'' if cumulative else '_new'}" if death else f"positive{'' if cumulative else '_new'}_cases"}_count''' 
    cdfs, columns = [], []
    if county_level:
        if selected_counties is None:
            epi_df = epi_df.loc[(epi_df[ctry_col]=='United States') & (epi_df[state_col].isin(state2abbr.keys())) & (epi_df[county_col]!='Unknown'), 
                                [date_col, state_col, county_col, case_col]]
        else:
            states, counties = zip(*[c.split('/') for c in selected_counties])
            selected_states = list(set(states))
            selected_counties = list(set(counties))
            epi_df = epi_df.loc[(epi_df[ctry_col]=='United States') & (epi_df[state_col].isin(selected_states)) & (epi_df[county_col].isin(selected_counties)), 
                                [date_col, state_col, county_col, case_col]]
        for (state, county), data in epi_df.groupby([state_col, county_col]):
            data = data.loc[:, [date_col, case_col]].groupby(date_col).sum()
            data.index = pd.to_datetime(data.index)
            cdfs.append(data)
            columns.append(f"{state}/{county}")
    else:
        epi_df = epi_df.loc[(epi_df[ctry_col]=='United States') & (epi_df[state_col].isin(state2abbr.keys())), 
                            [date_col, state_col, county_col, case_col]]
        for state, data in epi_df.groupby(state_col):
            data = data.loc[:, [date_col, case_col]].groupby(date_col).sum()
            cdfs.append(data)
            columns.append(state)
    
    epi_df = pd.concat(cdfs, axis=1)
    epi_df.columns = columns
    epi_df.index.name = 'date'
    epi_df.index = pd.to_datetime(epi_df.index)
    epi_df.fillna(0.0, inplace=True)
    start_date = pd.to_datetime(start_date)
    if end_date is None:
        end_date = pd.to_datetime(datetime.today().date())
    else:
        end_date = pd.to_datetime(end_date)
    end_date = end_date - pd.Timedelta(1, unit='d')
    epi_df = epi_df.loc[start_date:end_date]
    return epi_df



def load_world_covid_dataset(
    death: bool = False,
    cumulative: bool = True,
    n_ctry: Optional[int] = None,
    start_date: str = '2020-01-23',
    end_date: Optional[str] = None,
):
    
    epi_df = dw.load_dataset(
        dataset_key='covid-19-data-resource-hub/covid-19-case-counts',
        force_update=False,
        auto_update=True, 
    ).dataframes['covid_19_activity']
    ctry_col = 'country_short_name'
    date_col = 'report_date'
    case_col = f'''people_{f"death{'' if cumulative else '_new'}" if death else f"positive{'' if cumulative else '_new'}_cases"}_count''' 
    if n_ctry is None:
        ref_ctry = ref_countries
    else:
        ref_ctry = epi_df.loc[:[date_col, ctry_col, state_col, case_col]].groupby([ctry_col, date_col]).sum().reset_index()\
            .groupby(ctry_col).last().sort_values(case_col, ascending=False).head(n_ctry).index.values
    ref_data = epi_df.loc[epi_df[ctry_col].isin(ref_ctry), [date_col, ctry_col, case_col]]
    columns, cdfs = [], []
    for ctry, data in ref_data.groupby(ctry_col):
        data = data[[date_col, case_col]].groupby(date_col).sum()
        data.index = pd.to_datetime(data.index)
        cdfs.append(data)
        columns.append(ctry)
    epi_df = pd.concat(cdfs, axis=1)
    epi_df.columns = columns
    epi_df.index.name = 'date'
    epi_df.index = pd.to_datetime(epi_df.index)
    epi_df.fillna(0.0, inplace=True)
    start_date = pd.to_datetime(start_date)
    if end_date is None:
        end_date = pd.to_datetime(datetime.today().date())
    else:
        end_date = pd.to_datetime(end_date)
    end_date = end_date - pd.Timedelta(1, unit='d')
    epi_df = epi_df.loc[start_date:end_date]
    return epi_df



def load_hospitalized_data(
    currently: bool = False,
    increase: bool = False,
    cumulative: bool = True,
    start_date: str = '2020-01-23',
    end_date: Optional[str] = None,
    valid_states: Optional[List[str]] = hosp_states,
):
    assert int(cumulative) + int(increase) + int(currently) == 1, (
        f"One and only one option shall be enabled."
    )
    raw = pd.read_csv(
        'https://api.covidtracking.com/v1/states/daily.csv'
    ).rename(columns=str.lower)
    usecols = ['date', 'state']
    hosp_col = 'hospitalized{}'
    if currently:
        hosp_col = hosp_col.format('currently')
    elif increase:
        hosp_col = hosp_col.format('increase')
    elif cumulative:
        hosp_col = hosp_col.format('cumulative')
    usecols.append(hosp_col)
    hosp_df = raw.loc[:, usecols]
    hosp_df['date'] = pd.to_datetime(hosp_df['date'], format="%Y%m%d")
    hosp_df = hosp_df.set_index('date')
    hosp_df = hosp_df.sort_index()
    states = {}
    if valid_states is None:
        valid_states = list(abbr2state.keys())
    for state, data in hosp_df.groupby('state'):
        if state not in valid_states:
            continue
        states[state] = data.loc[:, hosp_col]
    hosp_df = pd.DataFrame(states)
    hosp_df = hosp_df.rename(columns=abbr2state)
    if end_date is not None:
        end_date = pd.to_datetime(end_date) - pd.Timedelta(1, unit='d')
    hosp_df = hosp_df.bfill().fillna(0.0)
    hosp_df = hosp_df.loc[start_date:end_date]
    return hosp_df    
    


def load_bed_and_population_data():
    beds = dw.load_dataset(
        dataset_key='liz-friedman/hospital-capacity-data-from-hghi',
        force_update=False,
        auto_update=True,
    ).dataframes['20_population']
    beds = beds.loc[:, [
        'hrr', 
        'total_hospital_beds', 
        'total_icu_beds', 
        'adult_population', 
        'population_65'
    ]]
    beds[['county', 'state']] = beds.hrr.str.split(', ', expand=True)
    beds = beds.loc[:, [
        'state', 
        'total_hospital_beds', 
        'total_icu_beds', 
        'adult_population', 
        'population_65'
    ]].groupby('state').sum()
    geo = pd.read_csv(
        "https://raw.githubusercontent.com/COVID19Tracking/associated-data/master/us_census_data/us_census_2018_population_estimates_states.csv",
        usecols=['state', 'population', 'pop_density'],
        index_col='state',
    )
    geo['area'] = geo['population'] / geo['pop_density']
    beds['density'] = beds['adult_population'] / geo.loc[beds.index, 'area']
    return beds

    
def load_mobility_data(
    start_date: str = '2020-01-23',
    end_date: Optional[str] = None,
):
    mob = pd.read_csv(
        "https://raw.githubusercontent.com/descarteslabs/DL-COVID-19/master/DL-us-mobility-daterow.csv",
        usecols=['date', 'admin_level', 'admin1', 'admin2', 'm50', 'm50_index'],
        parse_dates=['date'],
    )
    mob_state = mob.loc[mob['admin_level']==1, ['date', 'admin1', 'm50']]
    states, dfs = zip(*[(state, data.set_index('date').sort_index().loc[:,['m50']]) for state, data in mob_state.groupby('admin1')])
    mob = pd.concat(dfs, axis=1)
    states = list(states)
    states[states.index('Washington, D.C.')] = 'District of Columbia'
    mob.columns = states
    if end_date is not None:
        end_date = pd.to_datetime(end_date) - pd.Timedelta(1, unit='d')
    mob = mob.loc[start_date:end_date, list(state2abbr.keys())]
    return mob


def load_census_embedding():
    npz = np.load(pkg_root.joinpath('embeddings.npz'))
    feats = pd.DataFrame(
        npz['emb_lin'],
        index=npz['region_names'],
        columns=npz['feature_names'],
    )
    return feats