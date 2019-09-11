#!/usr/bin/env python3

import argparse
from math import exp
import random

import nflgame
import numpy as np
import pandas as pd

from nflmodel import model, data


nfl_spreads = model.MeloNFL.from_cache('spread')


def mcmc_sample(games, decay=100000, steps=100000):
    """
    Perform Markov-chain Monte Carlo to optimize future picks

    """
    weeks = games.columns

    # total expected score
    def total(picks):
        score = 0
        for dw, (team, week) in enumerate(zip(picks, weeks)):
            score += exp(-dw/decay) * games.at[team, week]

        return score

    # initialize random picks
    while True:
        picks = random.sample(games.index, weeks.size)
        if total(picks) != -float('inf'):
            break

    # monte carlo metropolis hastings
    for step in range(steps):

        # machine epsilon to pad division by zero
        TINY = 1e-12

        # annealing temperature
        x = step/float(steps)
        T = 20*(1. - x) + TINY

        # initialize new picks
        new_picks = list(picks)

        # choose random week and corresponding team
        i1 = random.randrange(weeks.size)
        team1 = picks[i1]

        # choose second team and swap if necessary
        team2 = random.choice(games.index)
        if team2 in picks:
            new_picks[picks.index(team2)] = team1

        # set first week's new team
        new_picks[i1] = team2

        # compute spreads
        ts = total(picks)
        new_ts = total(new_picks)

        # MCMC step
        # note: the "or" short-circuits so if new_ts > ts then the
        # exp() > random is not evaluated
        if new_ts > ts or exp((new_ts - ts)/T) > random.random():
            picks = new_picks

        yield picks


def simulate_season(season_year, decay=1000, steps=100000):
    """
    Simulates the NFL pickem game for an entire NFL season

    """
    def pick_gen(week=1, picked=[]):

        # exit generator
        if week > 17:
            raise StopIteration

        # prediction time
        games = nflgame.games(season_year, week=week, kind='REG')
        time = min([g.eid for g in games]).replace(tzinfo=None)

        # query all games
        games = nflgame.games(
            season_type='Regular', season_year=season_year, week__ge=week)

        # model predictions
        predictions = pd.DataFrame(
            data=-float('inf'),
            index=nfl_spreads.labels,
            columns=range(week, 18)
        )

        # game outcomes
        outcomes = pd.DataFrame(
            data=0,
            index=nfl_spreads.labels,
            columns=range(week, 18)
        )

        # predict every game of the season
        for g in sorted(games, key=lambda g: g.start_time):
            home = g.home_team
            away = g.away_team

            pred_points = nfl_spreads.median(time, home, away)
            predictions.at[home, g.week] = pred_points
            predictions.at[away, g.week] = -pred_points

            points = g.home_score - g.away_score
            outcomes.at[home, g.week] = points
            outcomes.at[away, g.week] = -points

        print(predictions)

        # exclude teams that have been picked
        for pick in picked:
            predictions = predictions.drop(pick, axis=0, errors='ignore')

        # perform markov chain monte carlo
        for picks in mcmc_sample(predictions, decay=decay, steps=steps):
            best_picks = picks

        yield best_picks[0], outcomes.at[best_picks[0], week]

        # there's no yield from in python2
        for next_pick in pick_gen(week + 1, picked + best_picks[:1]):
            yield next_pick

    picks = list(pick_gen(week=1))
    teams, points = zip(*picks)
    print(teams, points)

    return teams, sum(points)

#simulate_season(2018, decay=1000, steps=100000)
#quit()


def loss(halflife):

    db = nfldb.connect()

    residuals = []

    for season_year in range(2010, 2019):
        q = nfldb.Query(db)
        q.game(season_type='Regular', season_year=season_year, finished=True)
        games = q.as_games()

        for pair in np.random.choice(games, size=(100000, 2), replace=True):
            prev, now = sorted(pair, key=lambda g: g.start_time)

            elapsed = (now.start_time - prev.start_time).total_seconds()
            elapsed /= (60 * 60 * 24 * 7)

            time = prev.start_time.replace(tzinfo=None)
            home = now.home_team
            away = now.away_team

            pred_spread = .5**(elapsed/halflife) * nfl_spreads.median(time, home, away)
            true_spread = now.home_score - now.away_score

            residuals.append(true_spread - pred_spread)

    return halflife, np.abs(residuals).mean()


if __name__ == "__main__":
    """
    Optimize 'Pickem' over an entire NFL season.
    Uses Markov chain Monte Carlo to optimze the expected score.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--picked",
            nargs='*',
            action="store",
            default=[],
            type=str,
            help="list of teams already picked"
            )
    parser.add_argument(
            "--season",
            action="store",
            default=2019,
            type=int,
            help="nfl season year"
            )
    parser.add_argument(
            "--steps",
            action="store",
            default=10**3,
            type=int,
            help="markov chain monte carlo steps")

    args = parser.parse_args()
    args_dict = vars(args)

    # parse args
    picks = args_dict['picked']
    season = args_dict['season']
    steps = args_dict['steps']

    # initialize pandas dataframe
    df = pd.DataFrame(
        data=-float('inf'),
        index=nfl_spreads.labels,
        columns=range(1, 18)
    )

    # predict every game of the season
    for g in nflgame._search_schedule(season, kind='REG'):
        eid = g['eid']
        week = g['week']
        home = g['home']
        away = g['away']

        date = '-'.join([eid[:4], eid[4:6], eid[6:8]])
        score = nfl_spreads.median(date, home, away)

        df.at[home, week] = score
        df.at[away, week] = -score

    # exclude teams that have been picked
    for week, pick in enumerate(picks, start=1):
        df = df.drop(pick, axis=0)
        df = df.drop(week, axis=1)

    # enter vegas lines in place of model
    # df.at['JAC', 15] = 7
    # df.at['WAS', 15] = -7

    # perform markov chain monte carlo
    for pick in mcmc_sample(df, steps=steps):
        mypicks = pick

    print(df)
    print(mypicks)

    score = sum([
        df.at[pick, week]
        for week, pick in enumerate(mypicks, start=1)
    ])
    print(score)
