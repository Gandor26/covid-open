from pathlib import Path
import json

pkg_root = Path(__file__).parent
with open(pkg_root.joinpath('us_state_abbr.json'), 'r') as f:
    state2abbr = json.load(f)
    abbr2state = {v:k for k,v in state2abbr.items()}

