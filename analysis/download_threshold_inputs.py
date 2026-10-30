"""Download the public assembly metric tables used by the stability analysis."""

import argparse
from pathlib import Path
from urllib.request import urlretrieve


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('output', type=Path)
parser.add_argument('--species', nargs='+', required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
for species in args.species:
    for tier in ('pass', 'warn', 'fail'):
        filename = f'{species}_atb_{tier}.csv.gz'
        url = f'https://static.qualibact.org/static/species/{species}/qualibact-v1.0/{filename}'
        urlretrieve(url, args.output / filename)
        print(filename)
