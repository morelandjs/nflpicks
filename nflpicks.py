#!/usr/bin/env python2

import matplotlib.pyplot as plt
import numpy as np
import nflgame
import random
import h5py
import os
from itertools import chain
from math import exp

'''
Directions:

1) Enter the teams below which you've already picked this season
2) Delete ratings.hdf (caches rating data and should be regenerated every week)
4) Run the script ./nflpicks.py
'''

# teams picked
teams_picked = ['SEA', 'DET']
weeks_played = len(teams_picked)

# team schedule, '@' denotes away games
matchups = dict(
        ARI = ['NE', 'TB', '@BUF', 'LA', '@SF', 'NYJ', 'SEA', '@CAR', 'BYE',
              'SF', '@MIN', '@ATL', 'WAS', '@MIA', 'NO', '@SEA', '@LA'],
        ATL = ['TB', '@OAK', '@NO', 'CAR', '@DEN', '@SEA', 'SD', 'GB', '@TB',
              '@PHI', 'BYE', 'ARI', 'KC', '@LA', 'SF', '@CAR', 'NO'],
        BAL = ['BUF', '@CLE', '@JAC', 'OAK', 'WAS', '@NYG', '@NYJ', 'BYE',
              'PIT', 'CLE', '@DAL', 'CIN', 'MIA', '@NE', 'PHI', '@PIT', '@CIN'],
        BUF = ['@BAL', 'NYJ', 'ARI', '@NE', '@LA', 'SF', '@MIA', 'NE', '@SEA',
              'BYE', '@CIN', 'JAC', '@OAK', 'PIT', 'CLE', 'MIA', '@NYJ'],
        CAR = ['@DEN', 'SF', 'MIN', '@ATL', 'TB', '@NO', 'BYE', 'ARI', '@LA',
              'KC', 'NO', '@OAK', '@SEA', 'SD', '@WAS', 'ATL', '@TB'],
        CHI = ['@HOU', 'PHI', '@DAL', 'DET', '@IND', 'JAC', '@GB', 'MIN',
              'BYE', '@TB', '@NYG', 'TEN', 'SF', '@DET', 'GB', 'WAS', '@MIN'],
        CIN = ['@NYJ', '@PIT', 'DEN', 'MIA', '@DAL', '@NE', 'CLE', 'WAS',
              'BYE', '@NYG', 'BUF', '@BAL', 'PHI', '@CLE', 'PIT', '@HOU', 'BAL'],
        CLE = ['@PHI', 'BAL', '@MIA', '@WAS', 'NE', '@TEN', '@CIN', 'NYJ',
              'DAL', '@BAL', 'PIT', 'NYG', 'BYE', 'CIN', '@BUF', 'SD', '@PIT'],
        DAL = ['NYG', '@WAS', 'CHI', '@SF', 'CIN', '@GB', 'BYE', 'PHI',
              '@CLE', '@PIT', 'BAL', 'WAS', '@MIN', '@NYG', 'TB', 'DET', '@PHI'],
        DEN = ['CAR', 'IND', '@CIN', '@TB', 'ATL', '@SD', 'HOU', 'SD', '@OAK',
              '@NO', 'BYE', 'KC', '@JAC', '@TEN', 'NE', '@KC', 'OAK'],
        DET = ['@IND', 'TEN', '@GB', '@CHI', 'PHI', 'LA', 'WAS', '@HOU',
              '@MIN', 'BYE', 'JAC', 'MIN', '@NO', 'CHI', '@NYG', '@DAL', 'GB'],
        GB  = ['@JAC', '@MIN', 'DET', 'BYE', 'NYG', 'DAL', 'CHI', '@ATL',
              'IND', '@TEN', '@WAS', '@PHI', 'HOU', 'SEA', '@CHI', 'MIN', '@DET'],
        HOU = ['CHI', 'KC', '@NE', 'TEN', '@MIN', 'IND', '@DEN', 'DET', 'BYE',
              '@JAC', '@OAK', 'SD', '@GB', '@IND', 'JAC', 'CIN', '@TEN'],
        IND = ['DET', '@DEN', 'SD', '@JAC', 'CHI', '@HOU', '@TEN', 'KC',
              '@GB', 'BYE', 'TEN', 'PIT', '@NYJ', 'HOU', '@MIN', '@OAK', 'JAC'],
        JAC = ['GB', '@SD', 'BAL', 'IND', 'BYE', '@CHI', 'OAK', '@TEN', '@KC',
              'HOU', '@DET', '@BUF', 'DEN', 'MIN', '@HOU', 'TEN', '@IND'],
        KC  = ['SD', '@HOU', 'NYJ', '@PIT', 'BYE', '@OAK', 'NO', '@IND',
              'JAC', '@CAR', 'TB', '@DEN', '@ATL', 'OAK', 'TEN', 'DEN', '@SD'],
        LA  = ['@SF', 'SEA', '@TB', '@ARI', 'BUF', '@DET', 'NYG', 'BYE',
              'CAR', '@NYJ', 'MIA', '@NO', '@NE', 'ATL', '@SEA', 'SF', 'ARI'],
        MIA = ['@SEA', '@NE', 'CLE', '@CIN', 'TEN', 'PIT', 'BUF', 'BYE',
              'NYJ', '@SD', '@LA', 'SF', '@BAL', 'ARI', '@NYJ', '@BUF', 'NE'],
        MIN = ['@TEN', 'GB', '@CAR', 'NYG', 'HOU', 'BYE', '@PHI', '@CHI',
              'DET', '@WAS', 'ARI', '@DET', 'DAL', '@JAC', 'IND', '@GB', 'CHI'],
        NE  = ['@ARI', 'MIA', 'HOU', 'BUF', '@CLE', 'CIN', '@PIT', '@BUF',
              'BYE', 'SEA', '@SF', '@NYJ', 'LA', 'BAL', '@DEN', 'NYJ', '@MIA'],
        NO  = ['OAK', '@NYG', 'ATL', '@SD', 'BYE', 'CAR', '@KC', 'SEA', '@SF',
              'DEN', '@CAR', 'LA', 'DET', '@TB', '@ARI', 'TB', '@ATL'],
        NYG = ['@DAL', 'NO', 'WAS', '@MIN', '@GB', 'BAL', '@LA', 'BYE', 'PHI',
              'CIN', 'CHI', '@CLE', '@PIT', 'DAL', 'DET', '@PHI', '@WAS'],
        NYJ = ['CIN', '@BUF', '@KC', 'SEA', '@PIT', '@ARI', 'BAL', '@CLE',
              '@MIA', 'LA', 'BYE', 'NE', 'IND', '@SF', 'MIA', '@NE', 'BUF'],
        OAK = ['@NO', 'ATL', '@TEN', '@BAL', 'SD', 'KC', '@JAC', '@TB', 'DEN',
              'BYE', 'HOU', 'CAR', 'BUF', '@KC', '@SD', 'IND', '@DEN'],
        PHI = ['CLE', '@CHI', 'PIT', 'BYE', '@DET', '@WAS', 'MIN', '@DAL',
              '@NYG', 'ATL', '@SEA', 'GB', '@CIN', 'WAS', '@BAL', 'NYG', 'DAL'],
        PIT = ['@WAS', 'CIN', '@PHI', 'KC', 'NYJ', '@MIA', 'NE', 'BYE', '@BAL',
              'DAL', '@CLE', '@IND', 'NYG', '@BUF', '@CIN', 'BAL', 'CLE'],
        SD  = ['@KC', 'JAC', '@IND', 'NO', '@OAK', 'DEN', '@ATL', '@DEN',
              'TEN', 'MIA', 'BYE', '@HOU', 'TB', '@CAR', 'OAK', '@CLE', 'KC'],
        SEA = ['MIA', '@LA', 'SF', '@NYJ', 'BYE', 'ATL', '@ARI', '@NO', 'BUF',
              '@NE', 'PHI', '@TB', 'CAR', '@GB', 'LA', 'ARI', '@SF'],
        SF  = ['LA', '@CAR', '@SEA', 'DAL', 'ARI', '@BUF', 'TB', 'BYE', 'NO',
              '@ARI', 'NE', '@MIA', '@CHI', 'NYJ', '@ATL', '@LA', 'SEA'],
        TB  = ['@ATL', '@ARI', 'LA', 'DEN', '@CAR', 'BYE', '@SF', 'OAK',
              'ATL', 'CHI', '@KC', 'SEA', '@SD', 'NO', '@DAL', '@NO', 'CAR'],
        TEN = ['MIN', '@DET', 'OAK', '@HOU', '@MIA', 'CLE', 'IND', 'JAC',
              '@SD', 'GB', '@IND', '@CHI', 'BYE', 'DEN', '@KC', '@JAC', 'HOU'],
        WAS = ['PIT', 'DAL', '@NYG', 'CLE', '@BAL', 'PHI', '@DET', '@CIN',
              'BYE', 'MIN', 'GB', '@DAL', '@ARI', '@PHI', 'CAR', '@CHI', 'NYG'],
)
teams = list(matchups)
teams_avail = list(set(teams) - set(teams_picked))
nteams, nweeks = len(teams), len(matchups[teams[0]])


# get the score for a given game
def score(g, team):
    if g.is_home(team):
        return g.score_home, g.score_away
    else:
        return g.score_away, g.score_home


# store team score histories in an hdf5 file
def cache_ratings():
    with h5py.File('ratings.hdf', 'w') as f:
        #last5years = nflgame.games([2010, 2011, 2012, 2013, 2014, 2015, 2016])
        #hca = np.mean([g.score_home - g.score_away for g in last5years]) 
        hca = 2.476

        #Rtg_avg = np.array([[g.score_home, g.score_away] for g in
        #    chain(nflgame.games(2015), nflgame.games(2016))]).mean()
        Rtg_avg = 22.781
        
        for team in teams:
            if team == 'LA':
                team_old, team_new = 'STL', 'LA'
            elif team == 'JAC':
                team_old, team_new = 'JAC', 'JAX'
            else:
                team_old, team_new = team, team

            scores = np.array([score(g, team) for g in
                    chain(nflgame.games(2015, home=team_old, away=team_old),
                    nflgame.games(2016, home=team_new, away=team_new))])

            weights = np.exp(-np.arange(len(scores))[::-1]/6.)
            ORtg = np.average(scores[:,0], weights=weights) - Rtg_avg
            DRtg = np.average(scores[:,1], weights=weights) - Rtg_avg
            dset = f.create_dataset(team, data=[ORtg, DRtg])
            dset.attrs['hca'] = hca


# calculate offensive and defensive rating
def rating(team):
    if not os.path.exists('ratings.hdf'):
        cache_ratings()
    with h5py.File('ratings.hdf', 'r') as f:
        ORtg, DRtg = f[team]
        hca = f[team].attrs['hca']
        return ORtg, DRtg, hca


# approximate spread from offensive and defensive ratings
def plus_minus(team):
    team_off, team_def, hca = rating(team)
    spread = []

    for opp in matchups[team]:
        if opp == 'BYE':
            spread.append(-float('inf'))
        else:
            if '@' in opp:
                opp = opp.replace('@', '')
                diff = -hca
            else:
                diff = 0

            opp_off, opp_def, hca = rating(opp)
            diff += team_off - team_def - opp_off + opp_def
            spread.append(diff)

    return spread


# generate predicted spreads (plus-minus) for every game
spreads = {team : plus_minus(team) for team in teams}


# calculate total expected spread for a set of picks
def total_spread(picks):
    return sum(spreads[team][week + weeks_played] for week, team in enumerate(picks))


# picks generator
def make_picks(npicks=1000):
    # initialize random picks
    while True:
        picks = random.sample(teams_avail, nweeks - weeks_played)
        if total_spread(picks) != -float('inf'):  
            break

    for step in range(npicks):
        x = float(step)/float(npicks)
        T = (1. - x)*5.
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
    # repeat the MCMC simulation using a large number of steps each time
    for i, picks in enumerate(make_picks(int(1e4))):
        mypicks = picks 
        #plt.scatter(i, total_spread(picks))
    #plt.show()

    # print predictions
    for week, team in enumerate(mypicks):
        week += weeks_played
        team = nflgame.standard_team(team)
        opp = matchups[team][week]
        print "".join(entry.ljust(6) for entry in
                [team, opp, str(spreads[team][week])])

    # print total expected score
    print(total_spread(mypicks))

if __name__ == "__main__":
    main()
