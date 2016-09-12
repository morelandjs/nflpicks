This script uses a Metropolis-Hastings algorithm to optimize a set of picks for the pick'em NFL football game.
The pick'em game has a simple set of rules:

  1) Each week you will choose 1 team that you think will win their game by the most points.
  
  2) However, you can only pick each team once this season.

  3) You gain or lose points based on the point differential of the game and team you chose for that week.

  4) The user with the most points at the end of the season wins.

  5) Superlative awards will also be given out for the largest win and loss throughout the season.

  6) Your pick must be submitted by the start of the first game each week. No late picks.

  7) You may edit your pick as many times as you wish prior to the start of the first game.

  8) Picks will only become visible to other players after the start of the first game that week.

  9) If you do not submit a pick for a week, you will lose 7 points off of your score.

The algorithm involves two separate projections. First, how is each team expected to matchup against their opponents each week (predicted spread), and second—given knowledge of future spreads—what is the optimal set of unique picks which optimizes the expected total spread?

To project team spreads, the script calculates points scored and points allowed for each team averaged over the previous six games. These offensive and defensive ratings are then used to create a simple spread estimate for a given matchup.

Once the spreads are calculated for all games in the season, a Metropolis-Hastings algorithm is used to optimize the set of team picks. The algorithm first chooses a random week, and then considers changing the team chosen for that week to a random replacement. If this chosen replacement is a team which has already been chosen, the two picks are swapped. Candidate picks which improve the overall expected spread are accepted with 100% probability and candidate picks which lower the expected spread are accepted with a reduced proability, P ~ exp(-(score1 - score2)/T), where T is an effective noise parameter or 'temperature'. As the simulation progresses, the temperature is cooled to reduce noise and settle into the basin of the overall optimal solution.

The code returns a set of weekly picks and corresponding opponents for each week. It also provides an estimate of the expected score.
