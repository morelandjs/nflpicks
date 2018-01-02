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
    def __init__(self, mypicks=[], season=2017, mcmc_steps=10**6):
        self.year = season
        self.mcmc_steps = mcmc_steps
        self.next_week = len(mypicks) + 1

        self.nfldb = nfldb.connect()
        self.teams = self.league(season)
        self.spreads = self.game_spreads()


    def score(self, team, year, week):
        """
        Return the score for a team in given year, week

        """
        team = nfldb.standard_team(team)

        q = nfldb.Query(self.nfldb)
        q.game(season_type='Regular', season_year=year, week=week,
                finished=True, team=team)

        for g in q.as_games():
            points = g.home_score - g.away_score
            return points if team == g.home_team else -points

        return 0

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

    def game_spreads(self):
        """
        Predict the spreads for all future matchups in the season.
        Returns a nested Python dictionary, e.g. spreads[week][team] = -7

        """
        # calculate spreads using margin-dependent ELO library
        rating = melo.Rating()
       
        # initialize spreads
        spreads = defaultdict(
                lambda: defaultdict(
                    lambda: {'pred': -float('inf'), 'obs': 0}
                    )
                )

        # query all remaining games in the present season
        q = nfldb.Query(self.nfldb)
        q.game(season_year=self.year, week__ge=self.next_week,
                season_type='Regular')

        # loop over games and calculate the predicted spread
        for g in q.as_games():

            # game year and week
            year = g.season_year
            week = g.week

            # home team and away team
            home = g.home_team
            away = g.away_team

            # save observed spreads
            if g.finished:
                spreads[week][home]['obs'] = rating.points(g) 
                spreads[week][away]['obs'] = -rating.points(g)

            # predict spread using margin-dependent ELO
            spread = rating.predict_score(
                    home, away, year, week
                    )

            # save predicted spreads
            spreads[week][home]['pred'] = spread
            spreads[week][away]['pred'] = -spread

        return spreads

    def total_spread(self, picks):
        """
        Sum the projected spread for all picks

        """
        points = sum(self.spreads[week][team]['pred']
                for week, team in enumerate(picks, start=self.next_week))

        total = 0
        for week, team in enumerate(picks, start=self.next_week):
            points = self.spreads[week][team]['pred']
            if week == 17:
                points *= 0.6
            total += points

        return total

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

            yield list(enumerate(picks, start=self.next_week))

    def output(self, picks):
        """
        Nicely format picks and print to standard out

        """

        # weeks and teams
        weeks, teams = zip(*picks)

        # aggregate spread data for each pick
        spreads = [self.spreads[week][team] for (week, team) in picks]
        pred = [int(s['pred']) for s in spreads]

        # rating instance for game scores
        obs = [self.score(team, self.year, week) for (week, team) in picks]
        
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
        obs = '{}:'.format(self.year) + fmt_spreads.format(*obs) + tot_obs

        # print to stdout
        for text in week, team, pred, obs:
            print(text)


def main():
    """
    Define command line arguments. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--mypicks",
            nargs='*',
            action="store",
            default=[],
            type=str,
            help="list of team picks"
            )
    parser.add_argument(
            "--season",
            action="store",
            default=2017,
            type=int,
            help="NFL season year"
            )
    parser.add_argument(
            "--mcmc-steps",
            action="store",
            default=10**6,
            type=int,
            help="markov chain monte carlo steps")

    args = parser.parse_args()
    args_dict = vars(args)

    """
    Construct a new Pickem season and simulate the season.
    """
    pickem = Pickem(**args_dict)

    for pick in pickem.make_picks(mypicks=args_dict['mypicks']):
        mypicks = pick

    pickem.output(mypicks)


if __name__ == "__main__":
    main()
