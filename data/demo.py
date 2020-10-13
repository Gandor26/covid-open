from pathlib import Path
import pandas as pd
import numpy as np

from .constants import pkg_root, state2abbr

def load_demograph_data():
    race = pd.read_csv(
        pkg_root.joinpath('race.tsv'),
        sep='\t',
        thousands=',',
        index_col=0,
    )
    race.columns = [
        'population',
        'hispanic',
        'white',
        'black',
        'asian',
        'native',
    ]
    for col in race.columns[1:]:
        race[col] = race[col] / race['population']
    race = race.loc[list(state2abbr.keys())]

    gender = pd.read_csv(
        pkg_root.joinpath('gender.tsv'),
        sep='\t',
        index_col=0,
    ) 
    gender = gender[gender.columns[:-1]]
    gender.columns = ['male', 'female']
    gender = gender.loc[list(state2abbr.keys())]
    for col in gender.columns:
        gender[col] = gender[col] / race['population']

    demo = pd.concat([race, gender], axis=1)
    return demo