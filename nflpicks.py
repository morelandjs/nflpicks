#!/usr/bin/env python2

import numpy as np
import nfldb
import random
from collections import defaultdict
from math import exp

import melo

nweeks = 17
nested_dict = lambda: defaultdict(nested_dict)

class Pickem:
    def __init__(self, rating):
        self.rtg = rating
        self.spreads = nested_dict()

    def predict_spreads(self, year, week):
        q = nfldb.Query(self.nfldb)
        q.game(season_type='Regular', season_year=year, week=week)

        spreads = {}

        for game in q.as_games:
            home = game.home_team
            away = game.away_team

            spread = self.rtg.predict_score(home, away, year, week)

            spreads.update({home: spread})
            spreads.update({away: -spread})



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

def main():
    # global meta data
    npicks = int(2e6)

    # list of picks
    teams_picked = []

    # create melo rating instance
    rating = Rating(obs='score')

    # simulate a full season
    for week in np.arange(1, 18):
        weeks_left = np.arange(week, 18)

        # calculate spread ratings
        rtg = {team : rating.predict_score(team, opp, YEAR, week)
                for team in teams}

        # power rankings
        power_rankings(rtg)

        # generate predicted spreads (plus-minus) for every remaining game
        spreads = {team : [spread(team, w, rtg) for w in weeks_left]
                for team in teams}

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
