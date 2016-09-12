#!/usr/bin/env python2

import matplotlib.pyplot as plt
import numpy as np
import random
import nflgame
from math import exp

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

# measure home field advantage
last5years = nflgame.games([2010, 2011, 2012, 2013, 2014, 2015])
hca = np.mean([g.score_home - g.score_away for g in last5years]) 

# calculate offensive and defensive rating
def rating(team):
    try:
        home_games = nflgame.games(2015, home=team) 
        away_games = nflgame.games(2015, away=team) 
    except TypeError:
        team = 'STL'
        home_games = nflgame.games(2015, home=team) 
        away_games = nflgame.games(2015, away=team) 

    ORtg = np.mean([g.score_home for g in home_games]
            + [g.score_away for g in away_games])
    DRtg = np.mean([g.score_away for g in home_games]
            + [g.score_home for g in away_games])

    return ORtg, DRtg

# approximate spread from offensive and defensive ratings
def plus_minus(team):
    team_off, team_def = rating(team)
    score = []
    for opp in matchups[team]:
        if opp == 'BYE':
            score.append(-float('inf'))
        else:
            if '@' in opp:
                opp = opp.replace('@', '')
                adv = -hca
            else:
                adv = hca
            opp_off, opp_def = rating(opp)
            score.append((team_off + opp_def - opp_off
                    - team_def + adv)/2)

    return score

# generate predicted spreads (plus-minus) for every game
spreads = dict(
    patriots    = plus_minus('NE'),
    bills       = plus_minus('BUF'),
    dolphins    = plus_minus('MIA'),
    jets        = plus_minus('NYJ'),
    bengals     = plus_minus('CIN'),
    steelers    = plus_minus('PIT'),
    ravens      = plus_minus('BAL'),
    browns      = plus_minus('CLE'),
    colts       = plus_minus('IND'),
    texans      = plus_minus('HOU'),
    titans      = plus_minus('TEN'),
    jaguars     = plus_minus('JAC'),
    broncos     = plus_minus('DEN'),
    chargers    = plus_minus('SD'),
    chiefs      = plus_minus('KC'),
    raiders     = plus_minus('OAK'),
    cowboys     = plus_minus('DAL'),
    giants      = plus_minus('NYG'),
    eagles      = plus_minus('PHI'),
    redskins    = plus_minus('WAS'),
    packers     = plus_minus('GB'),
    vikings     = plus_minus('MIN'),
    lions       = plus_minus('DET'),
    bears       = plus_minus('CHI'),
    saints      = plus_minus('NO'),
    falcons     = plus_minus('ATL'),
    panthers    = plus_minus('CAR'),
    buccaneers  = plus_minus('TB'),
    seahawks    = plus_minus('SEA'),
    rams        = plus_minus('LA'),
    niners      = plus_minus('SF'),
    cardinals   = plus_minus('ARI'),
)

teams = list(spreads)
nteams = len(teams)
nweeks = 17

# calculate total expected spread for a set of picks
def total_spread(picks):
    return sum(spreads[team][week] for week, team in enumerate(picks))


# picks generator
def make_picks(npicks=1000):
    # initialize random picks
    while True:
        picks = random.sample(teams, nweeks)
        if total_spread(picks) != -float('inf'):  
            break

    for step in range(npicks):
        x = float(step)/float(npicks)
        T = (1. - x)*5.
        new_picks = list(picks)

        # choose random week and corresponding team
        i1 = random.randrange(nweeks)
        team1 = picks[i1]

        # choose second team and swap if necessary
        team2 = random.choice(teams)
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
    for picks in make_picks(100000):
        mypicks = picks 

    # print predictions
    for week, team in enumerate(mypicks):
        team = nflgame.standard_team(team)
        opp = matchups[team][week]
        col_width = 5
        print "".join(entry.ljust(col_width) for entry in [team, opp])

    # print total expected score
    print(total_spread(mypicks))

if __name__ == "__main__":
    main()
