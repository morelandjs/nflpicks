Pickem
======

This script uses a [Metropolis-Hastings](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm) algorithm to optimize a set of picks for the pick'em NFL football game.
The pick'em game has a simple set of rules:

- Each week you will choose 1 team that you think will win their game by the most points.
- However, you can only pick each team once each season.
- You gain or lose points based on the point differential of the game and team you chose for each week.
- The player with the most points at the end of the season wins.

The algorithm involves two separate projections. First, the expected spread for each game for each week, and second—given knowledge of future spreads—what is the optimal set of unique picks which optimizes the expected total spread?

To project team spreads, the script calculates points scored and points allowed for each team using a weighted average of previous games played (more recent games are given larger weights). These offensive and defensive ratings are then used to create a simple spread estimate for a given matchup.

Once the spreads are calculated for all games in the season, a Metropolis-Hastings algorithm is used to optimize the set of team picks. The algorithm first chooses a random week, and then considers changing the team chosen for that week to a random replacement. If this chosen replacement is a team which has already been chosen, the two picks are swapped. Candidate picks which improve the overall expected spread are accepted with 100% probability and candidate picks which lower the expected spread are accepted with a reduced proability, P ~ exp(-(score1 - score2)/T), where T is an effective noise parameter or 'temperature'. As the simulation progresses, the temperature is cooled to reduce noise and settle into the basin of the overall optimal solution.

The code returns a set of weekly picks and corresponding opponents for each week.
