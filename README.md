Pickem
======

This script uses a [Metropolis-Hastings](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm) algorithm to optimize a set of picks for the [NFL pick'em game](www.nflmargins.com).

The game has a simple set of rules:

- Each week you choose one team that you think will win their game by the most points.
- However, each team (pick) can only be used once each season.
- You gain or lose points based on the point differential of the game and team you chose each week.
- The player with the most points at the end of the season wins.

The algorithm involves two separate projections. First, it predicts the point spread for every remaining game of the season, and second—given this knowledge of future spreads—it determines the optimal set of unique picks, chosen to maximize the end-of-season score. The game spreads are projected using the margin-dependent Elo model ([melo](https://github.com/morelandjs/melo)). Once the spreads are predicted for all remaining games of the season, a Metropolis-Hastings algorithm is used to optimize the set of unique team picks. The algorithm first chooses a random week, and then considers changing the team chosen for that week with a random replacement. If this chosen replacement is a team which has already been chosen, the two picks are swapped. Candidate picks which improve the overall expected spread are accepted with 100% probability and candidate picks which lower the expected spread are accepted with a reduced proability, P ~ exp(-(score1 - score2)/T), where T is an effective noise parameter or 'temperature'. As the simulation progresses, the temperature is cooled to reduce noise and settle into the basin of the overall optimal solution. The tail of the MCMC chain thus represents an optimal set of picks.

Usage
-----

Simply enter the current season year and a list of previously picked teams
```
./pickem --season 2018 --picked KC LAR CAR
```

The code will return a list of optimal picks for all remaining games of the seasom
```
['JAC', 'NO', 'MIN', 'ATL', 'PIT', 'DAL', 'NYJ', 'LAC', 'BAL', 'PHI', 'WAS', 'CIN', 'NE', 'SEA']
```
