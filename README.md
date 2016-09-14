Requires python2.6, matplotlib, numpy, h5py and [nflgame](http://github.com/BurntSushi/nflgame) which can be installed with pip,

`pip install matplotlib numpy h5py nflgame`

This script uses a [Metropolis-Hastings](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm) algorithm to optimize a set of picks for the pick'em NFL football game.
The pick'em game has a simple set of rules:

- Each week you will choose 1 team that you think will win their game by the most points.
- However, you can only pick each team once this season.
- You gain or lose points based on the point differential of the game and team you chose for that week.
- The user with the most points at the end of the season wins.
- Each pick must be submitted by the start of the first game each week. No late picks.
- If you do not submit a pick for a week, you will lose 7 points off of your score.

The algorithm involves two separate projections. First, how each team is expected to match up against their opponents each week (predicted spread), and second—given knowledge of future spreads—what is the optimal set of unique picks which optimizes the expected total spread?

To project team spreads, the script calculates points scored and points allowed for each team averaged over the previous six games. These offensive and defensive ratings are then used to create a simple spread estimate for a given matchup.

Once the spreads are calculated for all games in the season, a Metropolis-Hastings algorithm is used to optimize the set of team picks. The algorithm first chooses a random week, and then considers changing the team chosen for that week to a random replacement. If this chosen replacement is a team which has already been chosen, the two picks are swapped. Candidate picks which improve the overall expected spread are accepted with 100% probability and candidate picks which lower the expected spread are accepted with a reduced proability, P ~ exp(-(score1 - score2)/T), where T is an effective noise parameter or 'temperature'. As the simulation progresses, the temperature is cooled to reduce noise and settle into the basin of the overall optimal solution.

The code returns a set of weekly picks and corresponding opponents for each week. It also provides an estimate of the expected score. You'll want to delete the scores.hdf cache file after each week and rerun the script to source the latest game score data. Read instructions in the script for further details. 

Track team scores: [http://pickem-points.herokuapp.com/](https://pickem-points.herokuapp.com/)

