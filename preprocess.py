import pandas as pd
import glob
import re
from itertools import product

populations = {
    'Japan':    126_800_000,
    'US'  :    327_200_000,
    'Germany' :  82_790_000,
    'Italy' :    60_480_000,
    'Spain' :    46_660_000,
    'Belgium' :  11_400_000,
    'Switzerland' : 8_570_000,
    'Iran' :     81_160_000,
    'Korea, South' : 51_470_000,
    'United Kingdom' : 66_440_000,
    'Netherlands' : 17_180_000,
        'France' : 66_990_000
}


fnames = glob.glob(r"..\COVID-19\csse_covid_19_data\csse_covid_19_daily_reports\*.csv")
if len(fnames) == 0:
    raise RuntimeError('You need the COVID-19 files (https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data)')


files = []

for f in fnames:
    m = re.match('.*([0-9]{2})-([0-9]{2})-([0-9]{4}).csv', f)

    if m != None:
        M, D, Y = m.group(1), m.group(2), m.group(3)
        files += [ ( (Y, M, D), f)]
    else:
        print('Ignore:', f)

countries = list(sorted(populations.keys()))
files = list(sorted(files))
labels = ['Deaths', 'Recovered', 'Confirmed']

dates = []
data_series = {}

for (c, l) in product(countries, labels):
    data_series[(c, l)] = []

for YMD, f in files:
    data = pd.read_csv(f)

    # There are different names for the country col ... sigh
    if 'Country_Region' in data.columns:
        country_col = 'Country_Region'
    elif 'Country/Region' in data.columns:
        country_col = 'Country/Region'
    else:
        raise Exception('Where is the country col?')

    data = data[ [country_col] + labels ]
    data = data.groupby([country_col]).sum().reset_index()
    mask = data[country_col].isin(countries)
    data = data[mask]
    
    dates += [ YMD ]
    for (c, l) in data_series:
        v = data[l][data[country_col] == c]
        if len(v) > 0:
            data_series[(c, l)] += [ v.values[0] ]
        else:
            data_series[(c, l)] += [ 0 ]
    

dates = [f'{Y}-{M}-{D}' for (Y, M, D) in dates]
data = pd.DataFrame(data_series, index=dates).transpose()
data.to_csv(r'data\COVID_series.csv')
