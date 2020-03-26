import random
import numpy as np

from matplotlib.dates import date2num
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker

import pandas as pd

from itertools import groupby

from matplotlib.dates import (WEEKLY, MO, DateFormatter,
                              rrulewrapper, RRuleLocator, drange)

# I chose 20 days of delay between rate of infection and outcones, based on this
# https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)30566-3/fulltext
DELAY = timedelta(days=20)

def small(x, tolerence = 0.0001):
    return abs(x) < tolerence

def bound(x, xmin, xmax):
    return min(max(x, xmin), xmax)

def convert_date(date_str):
    return date2num(datetime.strptime(date_str, '%Y-%m-%d') - DELAY)

def estimate(R, Dp, pop, CFR_covid=0.001, hospital_infection_mult = 1.0):
    alpha = - 0.15

    # Constants

    # Based on 0.01 prob of dying a year / 12 months / 1:10 change of being in ICU
    CFR_other =  0.01 / 100
    
    # Unknonws
    Rp = 100000
    prev = (Rp + Dp) / pop
    hospital_prev = 1 - (1 - prev)**hospital_infection_mult
    actual_DP = max(Dp - hospital_prev * CFR_other * pop, 1)
    f = R / Rp

    converge = False
    for i in range(1_000_000):

        # Equations defining the modelS

        # In hostpital infection is as if you had 'hospital_infection_mult'
        # independent chances of catching according to prevalence
        hospital_prev = 1 - (1 - prev)**hospital_infection_mult

        # We adjust Death to exclude those from other reasons with positive test
        # but not caused by positive test. 
        actual_DP = max(Dp - hospital_prev * (1 - CFR_covid)* CFR_other * pop, 1)

        # The prevalence is the number of resolved cases / population
        prev = (Rp + Dp) / pop

        # The testing rate of the non-dead population.
        # (We make an assumption that dead people are nearly all diagnosed
        # when positive with the virus).
        f = R / Rp
        
        # Based on CFR = D / R+D <=> R = D/cfr - D
        Rp_new = actual_DP / CFR_covid - Dp

        # Update Rp to convergence
        DRp = Rp - Rp_new           
        new_Rp = Rp + alpha * DRp
        Rp = bound(new_Rp, 1.0, pop - Dp)

        # Reduce the step size to get more precision
        if i % 10_000 == 0:
            alpha *= 0.9594847368473

            # Exit early if different is only small
            diffs = [ DRp / Rp ]
            raw_diffs = [ DRp ]
            if all(map(small, diffs)):
                converge = True
                break

    return (converge, i, [prev, CFR_covid, Rp, f, CFR_other])

# Raw numbers as of 23/March/2020
# format: (R, D, pop)
populations = {
	'Japan':    ( 235,   42, 126_800_000),
        'US' :     ( 295,  517, 327_200_000),
	'Germany' : ( 453,  123, 82_790_000),
	'Italy' :   (7432, 6077, 60_480_000),
	'Spain' :   (3355, 2207, 46_660_000),
	'Belgium' : ( 401,   88, 11_400_000),
	'Switzerland' : (131, 118, 8_570_000),
	'Iran' :    (8376, 1812, 81_160_000),
	'Korea, South' : (3166, 111, 51_470_000),
	'United Kingdom' : (135, 335, 66_440_000),
	'Netherlands' : (2, 213, 17_180_000),
        'France' : (2200, 860, 66_990_000)
}


def make_table(populations, CFR_covid, hospital_infection_mult, flx):
    print(f'Assumptions: COVID CFR: {100*CFR_covid:>5.2f}% In Hospital factor: x{hospital_infection_mult}', file=flx)
    print(f"{'Country':15}   Prev      CFR  Testing   Comorb.    Infected", file=flx)
    print('-'* 62)

    data = []
    for country in populations:
        R, Dp, pop = populations[country]
        converge, i, vals = estimate(R, Dp, pop, CFR_covid, hospital_infection_mult)
        cov = '* '[converge]
        prev, CFR_covid, Rp, f, CFR_other = vals

        # Compute comorbidity
        hospital_prev = 1 - (1 - prev)**hospital_infection_mult
        comorb = hospital_prev * CFR_other * pop / Dp

        print(f'{country:15} {100*prev:>5.2f}%   {100*CFR_covid:>5.2f}%   {100*f:>5.2f}%   {100*comorb:>5.2f}%    {int(Rp):9,d}', file=flx)

        
        # Checks
        cfr = Dp / (Rp + Dp)
        assert (small(cfr - (CFR_covid + prev * CFR_other), 0.01))
        assert (small(prev - ((Rp + Dp) / pop), 0.01))
        assert (small(cfr - (Dp / (Rp + Dp)), 0.01))
        assert (small(R -  f * Rp, 0.01))

def estimate_rate(pt0, pt1, upper_bound):
    dpt = pt1-pt0
    rate = ((dpt / (upper_bound - pt0)) + pt0)/pt0
    return rate

def estimate_project(R, D, CFR_covidx, hospital_infection_mult):
    upper_bound = 0.65
    
    prev_y = []
    for ri, di in zip(R, D):
        converge, i, vals = estimate(ri, di, pop, CFR_covidx, hospital_infection_mult)
        prev, CFR_covidx, Rp, f, CFR_other = vals
        prev_y += [ prev ]

    ## Plot a projection!
    rate = 0.50 * estimate_rate(prev_y[-2], prev_y[-1], upper_bound) \
         + 0.25 * estimate_rate(prev_y[-3], prev_y[-2], upper_bound) \
         + 0.25 * estimate_rate(prev_y[-4], prev_y[-3], upper_bound)
    assert rate > 0
    start_date = R.index[-1]
    start_date = datetime.strptime(start_date, '%Y-%m-%d') - DELAY
    start_prev = prev_y[-1]
    
    one_day = timedelta(days=1)
    proj_dates = [ start_date ]
    proj_prev  = [ prev ]

    for _ in range(60):
        start_prev += (rate * start_prev - start_prev) * (upper_bound - start_prev)
        start_date += one_day
        proj_dates += [ start_date]
        proj_prev  += [ start_prev ]

    Rindex = R.index

    return (Rindex, prev_y, proj_dates, proj_prev)

if __name__ == '__main__':
    # Print a table with all countries
    CFR_covid = 0.005
    hospital_infection_mult = 5

    with open(r'figures/table.txt', 'w') as flx:
        make_table(populations, CFR_covid, hospital_infection_mult, flx)

    fig = plt.figure(figsize=(14,7))
    ax = fig.add_subplot(111)


    consolidated_data = pd.read_csv(r'data/COVID_series.csv')
    dataset = []

    c = 0
    for country, data_rows in groupby(consolidated_data.iterrows(), lambda x: x[1][0]):
        Drow, Rrow, Crow = tuple(s for _, s in data_rows)

        S = Rrow[2:] + Drow[2:]

        if S[-1] > 10 and country in populations:
            pop = populations[country][-1]
            mask = (S > 50)
            
            R = Rrow[2:][mask]
            D = Drow[2:][mask]

            if not len(D) > 3:
                continue
            
            col = c % 10

            vals = estimate_project(R, D, CFR_covid, hospital_infection_mult)
            (Rindex, prev_y, proj_dates, proj_prev) = vals
            
            dates_x = list(map(convert_date, R.index))
            plt.plot_date(dates_x, prev_y, fmt='*-', label = country, color=f'C{col}')

            dates_proj = list(map(date2num, proj_dates))
            plt.plot_date(dates_proj, proj_prev, fmt='--', label = None, color=f'C{col}')

            vals_lo = estimate_project(R, D, CFR_covid/10.0, hospital_infection_mult)
            (_, prev_y_lo, _, proj_prev_lo) = vals_lo

            vals_hi = estimate_project(R, D, CFR_covid * 10, hospital_infection_mult)
            (_, prev_y_hi, _, proj_prev_hi) = vals_hi

            assert len(dates_x) == len(prev_y)
            assert len(dates_x) == len(prev_y_lo)
            assert len(dates_x) == len(prev_y_hi)
            
            dataset += [( dates_x, prev_y, prev_y_lo, prev_y_hi,
                          dates_proj, proj_prev, proj_prev_lo, proj_prev_hi,
                          f'C{col}', country , D)]

            # Update the color counter
            c += 1


    plt.axhline(y=0.65, color='r', linestyle='-', alpha=0.3)
    plt.axvline(x=date2num(datetime.today()), color='k', linestyle='-', alpha=0.3)

    plt.yscale('log')
    plt.legend(loc=2)

    rule = rrulewrapper(WEEKLY, byweekday=MO)
    loc = RRuleLocator(rule)
    formatter = DateFormatter('%y-%m-%d')

    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_tick_params(rotation=30, labelsize=10)

    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: f'{100*x:4.1f}%'))



    plt.ylabel('Prevalence, Estimated and Projected')
    plt.xlabel('Date')
    plt.title(f'COVID-19 Prevalence: Estimation & Projection (CFR Line {CFR_covid * 100}%)')

    plt.grid()
    plt.savefig(f'figures/All-prev.png', dpi=300)
    plt.close()

    # Make all country graphs
    for rec in dataset:
        ( dates_x, prev_y, prev_y_lo, prev_y_hi,
                          dates_proj, proj_prev, proj_prev_lo, proj_prev_hi,
                          col, country, D ) = rec
        print(country)

        fig = plt.figure(figsize=(14,7))
        ax = fig.add_subplot(111)
        
        plt.plot_date(dates_x, prev_y, fmt='*-', label = country, color=col)
        plt.plot_date(dates_proj, proj_prev, fmt='--', label = None, color=col)
        plt.fill_between(x=dates_x, y1=prev_y_lo, y2=prev_y_hi, alpha=0.4, color=col)
        plt.fill_between(x=dates_proj, y1=proj_prev_lo, y2=proj_prev_hi, alpha=0.2, color=col)

        # Plot actual deaths
        # plt.plot_date(dates_x, D, fmt='--', label = None, color='r')


        plt.axhline(y=0.65, color='r', linestyle='-', alpha=0.3)
        plt.axvline(x=date2num(datetime.today()), color='k', linestyle='-', alpha=0.3)

        plt.yscale('log')
        plt.legend(loc=2)

        rule = rrulewrapper(WEEKLY, byweekday=MO)
        loc = RRuleLocator(rule)
        formatter = DateFormatter('%y-%m-%d')

        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_tick_params(rotation=30, labelsize=10)

        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: f'{100*x:4.1f}%'))


        plt.ylabel('Prevalence, Estimated and Projected')
        plt.xlabel('Date')
        plt.title(f'{country} COVID-19 Prevalence: Estimation & Projection (CFR Line 0.1% [0.01%-1%])')

        plt.grid()
        plt.savefig(f'figures/{country}-prev.png', dpi=300)
        plt.close()


        
