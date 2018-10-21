#!/usr/bin/env python2

import argparse
from math import exp
import random

import pandas as pd
import nfldb

from melo_nfl import nfl_spreads


def mcmc_sample(games, steps=10000):
    """
    Perform Markov-chain Monte Carlo to optimize future picks

    """
    # total expected score
    def total(picks):
        return sum(df.at[t, w] for t, w in zip(picks, games.columns))

    # initialize random picks
    while True:
        picks = random.sample(games.index, games.columns.size)
        if total(picks) != -float('inf'):
            break

    # monte carlo metropolis hastings
    for step in range(steps):

        # machine epsilon to pad division by zero
        TINY = 1e-12

        # annealing temperature
        x = step/float(steps)
        T = 5*(1. - x) + TINY

        # initialize new picks
        new_picks = list(picks)

        # choose random week and corresponding team
        i1 = random.randrange(games.columns.size)
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
            default=2018,
            type=int,
            help="nfl season year"
            )
    parser.add_argument(
            "--steps",
            action="store",
            default=10**6,
            type=int,
            help="markov chain monte carlo steps")

    args = parser.parse_args()
    args_dict = vars(args)

    picks = args_dict['picked']
    season = args_dict['season']
    steps = args_dict['steps']

    # import nfl game data
    db = nfldb.connect()
    q = nfldb.Query(db)
    q.game(season_type='Regular', season_year=season)

    # initialize pandas dataframe
    df = pd.DataFrame(
        data=-float('inf'),
        index=nfl_spreads.labels,
        columns=range(1, 18)
    )

    # predict every game of the season
    for g in q.as_games():
        time = g.start_time.replace(tzinfo=None)
        week = g.week
        home = g.home_team
        away = g.away_team
        score = nfl_spreads.mean(time, home, away)

        df.at[home, week] = score
        df.at[away, week] = -score

    # exclude teams that have been picked
    for week, pick in enumerate(picks, start=1):
        df = df.drop(pick, axis=0)
        df = df.drop(week, axis=1)

    # perform markob chain monte carlo
    for pick in mcmc_sample(df, steps=steps):
        mypicks = pick

    print(df)
    print(mypicks)
