#!/usr/bin/env python2

import argparse
import random
from collections import defaultdict
from math import exp

import nfldb
import numpy as np
from tqdm import tqdm
from termcolor import colored

import melo


class Pickem:
    def __init__(self, season=2017, next_week=1, mcmc_steps=10**5):
        self.season = season
        self.next_week = next_week
        self.mcmc_steps = mcmc_steps
        self.nfldb = nfldb.connect()
        self.teams = self.league(season)
        self.spreads = self.project_spreads()

    def league(self, season):
        """
        Complete list of teams in the league for the given season.
        Note: all teams go by their current names, for example,
        STL = LAR, SD = LAC, etc.

        """
        q = nfldb.Query(self.nfldb)
        q.game(season_type='Regular', season_year=season)

        teams = set()

        for game in q.as_games():
            teams.update([game.home_team, game.away_team])

        return teams

    def project_spreads(self):
        """
        Predict the spreads for all future matchups in the season.
        Returns a nested Python dictionary, e.g. spreads[week][team] = -7

        """
        # calculate spreads using margin-dependent ELO library
        rating = melo.Rating(obs='score')
       
        # initialize spreads to -inf; used to avoid bye-weeks
        spreads = defaultdict(
                lambda: defaultdict(
                    lambda: -float('inf')
                    )
                )

        # query all remaining games in the present season
        q = nfldb.Query(self.nfldb)
        q.game(season_year=self.season, week__ge=self.next_week,
                season_type='Regular')

        # loop over games and calculate the predicted spread
        for game in q.as_games():

            # game year and week
            year = game.season_year
            week = game.week

            # home team and away team
            home = game.home_team
            away = game.away_team

            # predict spread using margin-dependent ELO
            spread = rating.predict_score(
                    home, away, year, week
                    )

            # save spreads for future look-up
            spreads[week][home] = spread
            spreads[week][away] = -spread

        return spreads

    def total_spread(self, picks):
        """
        Sum the projected spread for all picks

        """
        points = sum(
                self.spreads[week][team]
                for week, team in enumerate(picks, start=self.next_week)
                )

        return points

    def make_picks(self, mypicks=[]):
        """
        Perform Markov-chain Monte Carlo to optimize future picks

        """
        # teams that are available 
        teams_avail = list(set(self.teams) - set(mypicks))
        weeks_left = 18 - self.next_week

        # initialize random picks
        while True:
            picks = random.sample(teams_avail, weeks_left)
            if self.total_spread(picks) != -float('inf'):
                break

        # monte carlo metropolis hastings
        for step in range(self.mcmc_steps):

            # machine epsilon to pad division by zero
            TINY = 1e-12

            # annealing temperature
            x = step/float(self.mcmc_steps)
            T = 5*(1. - x) + TINY

            # initialize new picks
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
            ts = self.total_spread(picks)
            new_ts = self.total_spread(new_picks)

            # MCMC step
            # note: the "or" short-circuits so if new_ts > ts then the
            # exp() > random is not evaluated
            if new_ts > ts or exp((new_ts - ts)/T) > random.random():
                picks = new_picks

            yield picks

    def output(self, picks):
        # unpack data
        teams = picks
        weeks = np.arange(1, 18)
        #weeks, teams = zip(*picks)

        pred = [int(self.spreads[week][pick])
                for week, pick in enumerate(picks, start=1)]
        #obs = [int(self.observe(pick)) for pick in picks]
        obs = pred
        
        # formatting styles
        fmt_week = ' {:03d}' * len(weeks)
        fmt_picks = ' {:>3}' * len(teams)
        fmt_spreads = ' {:+03d}' * len(weeks)

        # end of season totals
        total = '{:>5}'.format('TOT')
        tot_pred = '{:>5}'.format(sum(pred))
        tot_obs = '{:>5}'.format(sum(obs))

        # print season pickem
        week = 'WEEK:' + fmt_week.format(*weeks)
        team = 'PICK:' + fmt_picks.format(*teams) + total
        pred = 'SPRD:' + fmt_spreads.format(*pred) + tot_pred
        obs = '{}:'.format(self.season) + fmt_spreads.format(*obs) + tot_obs

        # print to stdout
        for text in week, team, pred, obs:
            print(text)


def main():
    """
    Define command line arguments. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--season",
            action="store",
            default=2016,
            type=int,
            help="nfl season year"
            )
    parser.add_argument(
            "--mcmc-steps",
            action="store",
            default=10**5,
            type=int,
            help="markov chain monte carlo steps")

    args = parser.parse_args()
    args_dict = vars(args)

    """
    Construct a new Pickem season and simulate the season.
    """
    pickem = Pickem(**args_dict)
    for pick in pickem.make_picks():
        mypicks = pick

    pickem.output(mypicks)


if __name__ == "__main__":
    main()
