#!/usr/bin/env python2

import matplotlib.pyplot as plt
import numpy as np
import nflgame
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
dhist, dfut = 8., 34. 

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
nteams, nweeks = len(teams), len(matchups[teams[0]])


# home field advantage constant
try:
    hfa = pickle.load(open('.hfa.pkl', 'rb'))
except IOError:
    years = list(range(2014, 2017))
    hfa = np.mean([g.score_home - g.score_away
        for g in nflgame.games(years)]) 
    pickle.dump(hfa, open('.hfa.pkl', 'wb'))


# return the score and opponent for a single game
def score(team, game):
    diff = game.score_home - game.score_away - hfa
    if game.is_home(team):
        opp = game.away
    else:
        opp = game.home
        diff = -diff
    opp = opp.replace('JAC', 'JAX')
    opp = opp.replace('STL', 'LA')
    return diff, opp


# helper function to deal with naming quirks
def pull_games(years, team=None):
    if team is None:
        return nflgame.games(years)

    games = []

    for year in years:
        try:
            games += nflgame.games(year, home=team, away=team)
        except TypeError:
            team_ = team.replace('JAX', 'JAC').replace('LA', 'STL')
            games += nflgame.games(year, home=team_, away=team_)

    return games

# retrieve team game data 
def game_scores(team, years):
    games = pull_games(years, team=team)
    week = lambda w: (w % 16) + 1
    dates = [(game.season(), week(w))
            for w, game in enumerate(games)]
    scores = [score(team, game)
            for game in games]

    return zip(dates, scores)


# calculate offensive and defensive rating
def rating(team, year, week, opp_rtg=None):
    if not os.path.exists('scores.pkl'):
        years = [2015, 2016]
        scores = {team: game_scores(team, years)
                for team in teams}
        pickle.dump(scores, open('scores.pkl', 'wb'))
    with open('scores.pkl' , 'rb') as f:
        load = pickle.load(f)
        scores = [score for (yr, wk), score in load[team]
                if (yr < year or wk <= week)]

        # create time weights for exp decay
        time = np.arange(len(scores))[::-1]
        weights = np.exp(-time/dhist)

        # return weighted average
        try:
            diff = [spread + opp_rtg[opp] for (spread, opp) in scores]
        except TypeError:
            diff = [spread for (spread, opp) in scores]

        return np.average(diff, weights=weights)


# print current power rankings, ORtg - DRtg
def power_rankings(rtg):
    # sort by spread rating
    pwr_rnk = sorted([(team, rtg[team])
        for team in teams], key=lambda x: -x[1])

    # print power rankings
    print('\nPower Rankings:')
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


# approximate spread from offensive and defensive ratings
def plus_minus(team, rtg):
    spread = []
    for opp in matchups[team]:
        if opp == 'BYE':
            spread.append(-float('inf'))
        else:
            if '@' in opp:
                opp = opp.replace('@', '')
                spread.append(rtg[team] - rtg[opp] - hfa)
            else:
                spread.append(rtg[team] - rtg[opp] + hfa)
    return spread


# calculate total expected spread for a set of picks
def total_spread(picks, spreads):
    # standard deviation
    std_dev = lambda w: 1e-6 + w/5.

    # weeks played
    weeks_played = nweeks - len(picks)

    # return spread sum with information decay and random error
    means = [np.exp(-week/dfut) * spreads[team][week + weeks_played]
            for week, team in enumerate(picks)]

    # sample true spreads with error
    return sum([np.random.normal(loc=mean, scale=std_dev(week))
            for week, mean in enumerate(means)])


def weekly_spreads(rtg, weeks_played):
    print('\nWeekly Spreads:')
    for team in teams:
        opp = matchups[team][weeks_played]
        if '@' in opp or 'BYE' in opp or 'BYE' in team:
            continue
        spread = rtg[team] - rtg[opp] + hfa
        print("".join(entry.ljust(6) for entry in
            [team, opp, '{:.1f}'.format(spread)]))


def best_picks(counts, weeks_played, npicks):
    print('\nBest Pick\'em Picks:')
    ranked_picks = Counter(counts).most_common()
    for team, counts in ranked_picks:
        opp = matchups[team][weeks_played]
        percent = 100.*counts/(npicks/2)
        print "".join(entry.ljust(6) for entry in
                [team, opp, '{:.1f}%'.format(percent)])
    return ranked_picks[0][0]


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

        yield picks, ts


def main():
    # number of MCMC iterations
    npicks = int(1e6)

    # list of picks
    #teams_picked = ['SEA', 'DET', 'MIA', 'WAS', 'NE', 'BUF', 'CIN',
    #        'MIN', 'KC', 'ARI', 'PIT', 'NYG', 'DEN', 'ATL']
    teams_picked = []

    # simulate a full season
    for wk in np.arange(1, nweeks + 1):

        # set current week
        weeks_played = len(teams_picked)
        week = weeks_played + 1
        year = 2016

        # calculate spread ratings
        rtg = {team : rating(team, year, week) for team in teams}
        for it in range(5):
            rtg_ = {team : rating(team, year, week, rtg) for team in teams}
            rtg = rtg_

        # generate predicted spreads (plus-minus) for every game
        spreads = {team : plus_minus(team, rtg) for team in teams}

        # run the monte carlo
        counts = [picks[0] for it, (picks, ts)
                in enumerate(make_picks(teams_picked, spreads, npicks))
                if it > npicks/2]

        # power rankings
        power_rankings(rtg)

        # weekly spreads
        weekly_spreads(rtg, weeks_played)

        # print top three best picks
        teams_picked.append(best_picks(counts, weeks_played, npicks))

    # output final picks
    for week, team in enumerate(teams_picked):
        opp = matchups[team][week]
        print "".join(entry.ljust(6) for entry in [team, opp])


if __name__ == "__main__":
    main()
