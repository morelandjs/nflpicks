#!/usr/bin/env python2

import matplotlib.pyplot as plt
import numpy as np
import nflgame
import itertools
import random
import os
import pickle
from itertools import chain
from collections import Counter
from math import exp

'''
Directions:

1) Enter the teams below which you've already picked this season
2) Delete ratings.hdf (caches rating data and should be regenerated every week)
4) Run the script ./nflpicks.py
'''

# historical and future information decay
# e.g., np.exp(-games/dhist)
dhist = 6.

YEAR = 2016

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
nteams = len(teams)
nweeks = len(matchups['ARI'])
weeks = np.arange(nweeks)

# home field advantage constant
try:
    hfa = pickle.load(open('.hfa.pkl', 'rb'))
except IOError:
    years = list(range(2012, 2017))
    hfa = np.mean([g.score_home - g.score_away
        for g in nflgame.games(years)]) 
    pickle.dump(hfa, open('.hfa.pkl', 'wb'))


def games(year, **kwargs):
    replacements = dict(LA='STL', JAX='JAC')
    if isinstance(year, int):
        if year < 2016:
            for k in ['home', 'away']:
                if kwargs[k] in replacements:
                    kwargs[k] = replacements[kwargs[k]]
        return nflgame.games(year, **kwargs)

    return list(itertools.chain.from_iterable(
        games(y, **kwargs) for y in year
    ))


def opp(team, game):
    replacements = dict(STL='LA', JAC='JAX')
    opp = game.away if game.is_home(team) else game.home
    if opp in replacements:
        return replacements[opp]
    return opp
    

# return the score and opponent for a single game
def score(team, game, adv=0):
    diff = game.score_home - game.score_away
    spread = diff - adv if game.is_home(team) else -diff + adv
    return spread, opp(team, game)


# calculate spread rating for upcoming week
def rating(team, year, week, opp_rtg=None):
    years = [year - 1, year]
    scores = [score(team, g, hfa) for g in
            games(years, home=team, away=team)
            if g.schedule['week'] < week
            or g.season() < year]

    # create time weights for exp decay
    time = np.arange(len(scores))
    time[week:] += 4
    weights = np.exp(-time[::-1]/dhist)

    # return weighted average
    try:
        diff = [spread + opp_rtg[opp] for (spread, opp) in scores]
    except TypeError:
        diff = [spread for (spread, opp) in scores]
    return np.average(diff, weights=weights)


#def rating_adjusted(team, year, week):
#    rtg = {team : rating(team, year, week) for team in teams}
#    return {team : rating(team, year, week, rtg) for team in teams}


# approximate spread from offensive and defensive ratings
def spread(team, week, rtg):
    replacements = dict(STL='LA', JAC='JAX')
    if team in replacements:
        team = replacements[team]
    opp = matchups[team][week - 1]
    if opp == 'BYE':
        return -float('inf')
    if '@' in opp:
        opp = opp.replace('@', '')
        return rtg[team] - rtg[opp] - hfa
    else:
        return rtg[team] - rtg[opp] + hfa


# calculate total expected spread for a set of picks
def total_spread(picks, spreads):
    # standard deviation
    #std_dev = lambda w: 1e-6 + w/10.
    std_dev = lambda w: 1e-6

    # return spread sum with information decay and random error
    means = [np.arctan(spreads[team][week]/3.) for week, team in enumerate(picks)]

    # sample true spreads with error
    return sum([np.random.normal(loc=mean, scale=std_dev(week))
        for week, mean in enumerate(means)])


# picks generator
def make_picks(teams_picked, spreads, npicks=1000):
    # teams that are available 
    teams_avail = list(set(teams) - set(teams_picked))
    weeks_left = nweeks - len(teams_picked)

    # initialize random picks
    while True:
        picks = random.sample(teams_avail, weeks_left)
        if total_spread(picks, spreads) != -float('inf'):  
            break

    # monte carlo metropolis hastings
    for step in range(npicks):
        x = float(step)/float(npicks/2)
        T = max((1. - x)*5., 1e-12)
        new_picks = list(picks)

        # choose random week and corresponding team
        i1 = random.randrange(weeks_left)
        team1 = picks[i1]

        # choose second team and swap if necessary
        team2 = random.choice(teams_avail)
        if team2 in picks:
            new_picks[picks.index(team2)] = team1

        # set first week's new team
        new_picks[i1] = team2

        # compute spreads
        ts = total_spread(picks, spreads)
        new_ts = total_spread(new_picks, spreads)

        # MCMC step
        # note: the "or" short-circuits so if new_ts > ts then the
        # exp() > random is not evaluated
        if new_ts > ts or exp((new_ts - ts)/T) > random.random():
            picks = new_picks

        yield picks


# print current power rankings, ORtg - DRtg
def power_rankings(rtg):
    # sort by spread rating
    pwr_rnk = sorted([(team, rtg[team])
        for team in teams], key=lambda x: -x[1])

    # print power rankings
    print '\nPower Rankings:'
    for rnk in pwr_rnk:
        print "".join(str(entry).ljust(6) for entry in rnk)

    # plot power rankings
    team, rank = zip(*pwr_rnk)
    indices = np.arange(len(team))
    plt.bar(indices, rank, color=plt.cm.Blues(.6), lw=0)
    plt.xticks(indices + 0.4, team, rotation=90)
    plt.xlim(0, len(team))
    plt.ylabel('Average point differential')
    plt.savefig('ratings.pdf')


def predicted_spreads(rtg, week):
    print '\nPredicted Spreads:'
    games = nflgame.games(YEAR, week=week)
    for game in games:
        team = game.home.replace('JAC', 'JAX')
        pred_spread = spread(team, week, rtg)
        print "".join(entry.ljust(6) for entry in
            [team, opp(team, game), '{:.1f}'.format(pred_spread)])


def best_picks(team_samples, week):
    print '\nBest Pick\'em Picks:'
    counter = Counter(team_samples).most_common()
    for team, counts in counter:
        opp = matchups[team][week - 1]
        percent = 100.*counts/len(team_samples)
        print "".join(entry.ljust(6) for entry in
                [team, opp, '{:.1f}%'.format(percent)])

    return counter[0][0]


def projected_score(picks, rtg):
    _, this_week = nflgame.live.current_year_and_week()
    total = 0

    print '\nPick\'em Season Picks:' 
    for week, team in enumerate(picks, start=1):
        if week == this_week:
            break
        game = games(YEAR, week=week, home=team, away=team)
        diff, opp = score(team, *game)
        total += diff
        print "".join(entry.ljust(6)
                for entry in [team, opp,'{:.1f}'.format(diff)])
    print '\nTotal Score:', total

    print '\nProjected Remaining Picks:' 
    for week, team in enumerate(picks, start=1):
        if week < this_week:
            continue
        opp = matchups[team][week - 1]
        diff = spread(team, week, rtg)
        print "".join(entry.ljust(6)
                for entry in [team, opp,'{:.1f}'.format(diff)])


def main():
    # global meta data
    npicks = int(2e6)

    # list of picks
    teams_picked = []

    # simulate a full season
    for week in np.arange(1, 18):
        weeks_left = np.arange(week, 18)

        # calculate spread ratings
        rtg = {team : rating(team, YEAR, week) for team in teams}

        if week in [1, 17]:
            rtg = {team : 0.66*rtg[team] for team in teams}

        # power rankings
        power_rankings(rtg)

        # generate predicted spreads (plus-minus) for every remaining game
        spreads = {team : [spread(team, w, rtg) for w in weeks_left]
                for team in teams}

        # exclude injured teams
        if week == 5:
            spreads['CAR'][0] = 0

        # predicted spreads for each game
        predicted_spreads(rtg, week)

        # run the monte carlo
        team_samples = [picks[0] for it, picks in enumerate(
            make_picks(teams_picked, spreads, npicks)
            ) if it > npicks/2]

        # this week's matchups and spreads
        teams_picked.append(best_picks(team_samples, week))

    # output final picks
    projected_score(teams_picked, rtg)


if __name__ == "__main__":
    main()
