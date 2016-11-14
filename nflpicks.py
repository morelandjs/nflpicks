#!/usr/bin/env python2

import matplotlib.pyplot as plt
import numpy as np
import nflgame
import random
import h5py
import os
from itertools import chain
from collections import Counter
from math import exp

'''
Directions:

1) Enter the teams below which you've already picked this season
2) Delete ratings.hdf (caches rating data and should be regenerated every week)
4) Run the script ./nflpicks.py
'''

# teams picked
teams_picked = ['SEA', 'DET', 'MIA', 'WAS', 'NE', 'BUF', 'CIN', 'MIN', 'KC', 'ARI']
#teams_picked = []

# historical and future information decay
# e.g., np.exp(-games/dhist)
dhist, dfut = 6., 34. 

# team schedule, '@' denotes away games
matchups = dict(
        ARI = ['NE', 'TB', '@BUF', 'LA', '@SF', 'NYJ', 'SEA', '@CAR', 'BYE',
              'SF', '@MIN', '@ATL', 'WAS', '@MIA', 'NO', '@SEA', '@LA'],
        ATL = ['TB', '@OAK', '@NO', 'CAR', '@DEN', '@SEA', 'SD', 'GB', '@TB',
              '@PHI', 'BYE', 'ARI', 'KC', '@LA', 'SF', '@CAR', 'NO'],
        BAL = ['BUF', '@CLE', '@JAX', 'OAK', 'WAS', '@NYG', '@NYJ', 'BYE',
              'PIT', 'CLE', '@DAL', 'CIN', 'MIA', '@NE', 'PHI', '@PIT', '@CIN'],
        BUF = ['@BAL', 'NYJ', 'ARI', '@NE', '@LA', 'SF', '@MIA', 'NE', '@SEA',
              'BYE', '@CIN', 'JAX', '@OAK', 'PIT', 'CLE', 'MIA', '@NYJ'],
        CAR = ['@DEN', 'SF', 'MIN', '@ATL', 'TB', '@NO', 'BYE', 'ARI', '@LA',
              'KC', 'NO', '@OAK', '@SEA', 'SD', '@WAS', 'ATL', '@TB'],
        CHI = ['@HOU', 'PHI', '@DAL', 'DET', '@IND', 'JAX', '@GB', 'MIN',
              'BYE', '@TB', '@NYG', 'TEN', 'SF', '@DET', 'GB', 'WAS', '@MIN'],
        CIN = ['@NYJ', '@PIT', 'DEN', 'MIA', '@DAL', '@NE', 'CLE', 'WAS',
              'BYE', '@NYG', 'BUF', '@BAL', 'PHI', '@CLE', 'PIT', '@HOU', 'BAL'],
        CLE = ['@PHI', 'BAL', '@MIA', '@WAS', 'NE', '@TEN', '@CIN', 'NYJ',
              'DAL', '@BAL', 'PIT', 'NYG', 'BYE', 'CIN', '@BUF', 'SD', '@PIT'],
        DAL = ['NYG', '@WAS', 'CHI', '@SF', 'CIN', '@GB', 'BYE', 'PHI',
              '@CLE', '@PIT', 'BAL', 'WAS', '@MIN', '@NYG', 'TB', 'DET', '@PHI'],
        DEN = ['CAR', 'IND', '@CIN', '@TB', 'ATL', '@SD', 'HOU', 'SD', '@OAK',
              '@NO', 'BYE', 'KC', '@JAX', '@TEN', 'NE', '@KC', 'OAK'],
        DET = ['@IND', 'TEN', '@GB', '@CHI', 'PHI', 'LA', 'WAS', '@HOU',
              '@MIN', 'BYE', 'JAX', 'MIN', '@NO', 'CHI', '@NYG', '@DAL', 'GB'],
        GB  = ['@JAX', '@MIN', 'DET', 'BYE', 'NYG', 'DAL', 'CHI', '@ATL',
              'IND', '@TEN', '@WAS', '@PHI', 'HOU', 'SEA', '@CHI', 'MIN', '@DET'],
        HOU = ['CHI', 'KC', '@NE', 'TEN', '@MIN', 'IND', '@DEN', 'DET', 'BYE',
              '@JAX', '@OAK', 'SD', '@GB', '@IND', 'JAX', 'CIN', '@TEN'],
        IND = ['DET', '@DEN', 'SD', '@JAX', 'CHI', '@HOU', '@TEN', 'KC',
              '@GB', 'BYE', 'TEN', 'PIT', '@NYJ', 'HOU', '@MIN', '@OAK', 'JAX'],
        JAX = ['GB', '@SD', 'BAL', 'IND', 'BYE', '@CHI', 'OAK', '@TEN', '@KC',
              'HOU', '@DET', '@BUF', 'DEN', 'MIN', '@HOU', 'TEN', '@IND'],
        KC  = ['SD', '@HOU', 'NYJ', '@PIT', 'BYE', '@OAK', 'NO', '@IND',
              'JAX', '@CAR', 'TB', '@DEN', '@ATL', 'OAK', 'TEN', 'DEN', '@SD'],
        LA  = ['@SF', 'SEA', '@TB', '@ARI', 'BUF', '@DET', 'NYG', 'BYE',
              'CAR', '@NYJ', 'MIA', '@NO', '@NE', 'ATL', '@SEA', 'SF', 'ARI'],
        MIA = ['@SEA', '@NE', 'CLE', '@CIN', 'TEN', 'PIT', 'BUF', 'BYE',
              'NYJ', '@SD', '@LA', 'SF', '@BAL', 'ARI', '@NYJ', '@BUF', 'NE'],
        MIN = ['@TEN', 'GB', '@CAR', 'NYG', 'HOU', 'BYE', '@PHI', '@CHI',
              'DET', '@WAS', 'ARI', '@DET', 'DAL', '@JAX', 'IND', '@GB', 'CHI'],
        NE  = ['@ARI', 'MIA', 'HOU', 'BUF', '@CLE', 'CIN', '@PIT', '@BUF',
              'BYE', 'SEA', '@SF', '@NYJ', 'LA', 'BAL', '@DEN', 'NYJ', '@MIA'],
        NO  = ['OAK', '@NYG', 'ATL', '@SD', 'BYE', 'CAR', '@KC', 'SEA', '@SF',
              'DEN', '@CAR', 'LA', 'DET', '@TB', '@ARI', 'TB', '@ATL'],
        NYG = ['@DAL', 'NO', 'WAS', '@MIN', '@GB', 'BAL', '@LA', 'BYE', 'PHI',
              'CIN', 'CHI', '@CLE', '@PIT', 'DAL', 'DET', '@PHI', '@WAS'],
        NYJ = ['CIN', '@BUF', '@KC', 'SEA', '@PIT', '@ARI', 'BAL', '@CLE',
              '@MIA', 'LA', 'BYE', 'NE', 'IND', '@SF', 'MIA', '@NE', 'BUF'],
        OAK = ['@NO', 'ATL', '@TEN', '@BAL', 'SD', 'KC', '@JAX', '@TB', 'DEN',
              'BYE', 'HOU', 'CAR', 'BUF', '@KC', '@SD', 'IND', '@DEN'],
        PHI = ['CLE', '@CHI', 'PIT', 'BYE', '@DET', '@WAS', 'MIN', '@DAL',
              '@NYG', 'ATL', '@SEA', 'GB', '@CIN', 'WAS', '@BAL', 'NYG', 'DAL'],
        PIT = ['@WAS', 'CIN', '@PHI', 'KC', 'NYJ', '@MIA', 'NE', 'BYE', '@BAL',
              'DAL', '@CLE', '@IND', 'NYG', '@BUF', '@CIN', 'BAL', 'CLE'],
        SD  = ['@KC', 'JAX', '@IND', 'NO', '@OAK', 'DEN', '@ATL', '@DEN',
              'TEN', 'MIA', 'BYE', '@HOU', 'TB', '@CAR', 'OAK', '@CLE', 'KC'],
        SEA = ['MIA', '@LA', 'SF', '@NYJ', 'BYE', 'ATL', '@ARI', '@NO', 'BUF',
              '@NE', 'PHI', '@TB', 'CAR', '@GB', 'LA', 'ARI', '@SF'],
        SF  = ['LA', '@CAR', '@SEA', 'DAL', 'ARI', '@BUF', 'TB', 'BYE', 'NO',
              '@ARI', 'NE', '@MIA', '@CHI', 'NYJ', '@ATL', '@LA', 'SEA'],
        TB  = ['@ATL', '@ARI', 'LA', 'DEN', '@CAR', 'BYE', '@SF', 'OAK',
              'ATL', 'CHI', '@KC', 'SEA', '@SD', 'NO', '@DAL', '@NO', 'CAR'],
        TEN = ['MIN', '@DET', 'OAK', '@HOU', '@MIA', 'CLE', 'IND', 'JAX',
              '@SD', 'GB', '@IND', '@CHI', 'BYE', 'DEN', '@KC', '@JAX', 'HOU'],
        WAS = ['PIT', 'DAL', '@NYG', 'CLE', '@BAL', 'PHI', '@DET', '@CIN',
              'BYE', 'MIN', 'GB', '@DAL', '@ARI', '@PHI', 'CAR', '@CHI', 'NYG'],
)
teams = list(matchups)
teams_avail = list(set(teams) - set(teams_picked))
nteams, nweeks = len(teams), len(matchups[teams[0]])
weeks_played = len(teams_picked)


# get the score for a given game
def score(g, team):
    if g.is_home(team):
        return g.score_home, g.score_away
    else:
        return g.score_away, g.score_home


def games(years, team=None):
    if team is None:
        return nflgame.games(years)
    elif team is 'LA':
        oldteam = 'STL'
    elif team is 'JAX':
        oldteam = 'JAC'
    else:
        return nflgame.games(years, home=team, away=team)

    y1 = [y for y in years if y <  2016]
    y2 = [y for y in years if y >= 2016]

    for y, t in zip([y1, y2], [oldteam, team]):
        try:
            games += nflgame.games(y, home=t, away=t)
        except NameError:
            games = []

    return games


# store team score histories in an hdf5 file
def cache_scores():
    with h5py.File('scores.hdf', 'w') as f:
        years = list(range(2010, 2017))
        hca = np.mean([g.score_home - g.score_away
            for g in games(years)]) 

        for team in teams:
            scores = [score(g, team)
                    for g in games([2015, 2016], team=team)]
            dset = f.create_dataset(team, data=scores)
            dset.attrs['hca'] = hca


# calculate offensive and defensive rating
def rating(team):
    global hca
    if not os.path.exists('scores.hdf'):
        cache_scores()
    with h5py.File('scores.hdf', 'r') as f:
        scores = f[team]
        entries = len(scores)
        hca = f[team].attrs['hca']

        # create time weights for exp decay
        time = np.arange(entries)[::-1]
        time[:entries - weeks_played] += 4
        weights = np.exp(-time/dhist)

        # return weighted average
        ORtg, DRtg = np.average(scores, axis=0, weights=weights).T
        return ORtg - DRtg


# print current power rankings, ORtg - DRtg
def power_rankings():
    pwr_rnk = [(team, rating(team)) for team in teams]
    print('\nPower Rankings:')
    for pwr_rnk in sorted(pwr_rnk, key=lambda x: -x[1]):
        print "".join(str(entry).ljust(6) for entry in pwr_rnk)


# approximate spread from offensive and defensive ratings
def plus_minus(team):
    spread = []
    for opp in matchups[team]:
        if opp == 'BYE':
            spread.append(-float('inf'))
        else:
            if '@' in opp:
                opp = opp.replace('@', '')
                spread.append(rating(team) - rating(opp) - hca/2)
            else:
                spread.append(rating(team) - rating(opp) + hca/2)
    return spread


# generate predicted spreads (plus-minus) for every game
spreads = {team : plus_minus(team) for team in teams}

# make some tweaks due to injuries
#spreads['CAR'][4] -= 5

# calculate total expected spread for a set of picks
def total_spread(picks):
    # add a one touchdown error uncertainty to each predicted spread
    errors = np.random.normal(scale=7, size=len(picks))

    # return spread sum with information decay and random error
    return sum(np.exp(-week/dfut)*spreads[team][week + weeks_played] + error
            for week, (team, error) in enumerate(zip(picks, errors)))


# picks generator
def make_picks(npicks=1000):
    # initialize random picks
    while True:
        picks = random.sample(teams_avail, nweeks - weeks_played)
        if total_spread(picks) != -float('inf'):  
            break

    for step in range(npicks):
        x = float(step)/float(npicks/2)
        T = max((1. - x)*5., 1e-12)
        new_picks = list(picks)

        # choose random week and corresponding team
        i1 = random.randrange(nweeks - weeks_played)
        team1 = picks[i1]

        # choose second team and swap if necessary
        team2 = random.choice(teams_avail)
        if team2 in picks:
            new_picks[picks.index(team2)] = team1

        # set first week's new team
        new_picks[i1] = team2

        # compute spreads
        ts = total_spread(picks)
        new_ts = total_spread(new_picks)

        # MCMC step
        # note: the "or" short-circuits so if new_ts > ts then the
        # exp() > random is not evaluated
        if new_ts > ts or exp((new_ts - ts)/T) > random.random():
            picks = new_picks

        yield picks



def main():
    # number of MCMC iterations
    npicks = 2e6

    # record next week's pick occurences after initial burn in
    counts = [p[0] for (i, p) in enumerate(
        make_picks(int(npicks))) if i > npicks/2]

    # print current power rankings
    power_rankings()

    # plot next week's "best pick" likelihood
    labels, values = zip(*Counter(counts).most_common())
    indices = np.arange(len(labels))
    plt.bar(indices, values, color=plt.cm.Blues(.6), lw=0)
    plt.xticks(indices + 0.4, labels, rotation=90)
    plt.xlim(0, len(labels))
    plt.savefig('picks.pdf')

    # print top three best picks
    print('\nBest Pick\'em Picks:')
    for team, counts in Counter(counts).most_common():
        opp = matchups[team][weeks_played]
        percent = 100.*counts/(npicks/2)
        print "".join(entry.ljust(6) for entry in
                [team, opp, '{:.1f}%'.format(percent)])


if __name__ == "__main__":
    main()
