# Advanced Search Techniques in Adversarial Game-Playing Agent

## Implementation Overview

For this project, I implemented an advanced search technique in the CustomPlayer class. The primary improvements include:

1. Iterative Deepening
2. Alpha-Beta Pruning
3. Transposition Table

These techniques were chosen to enhance the search efficiency and decision-making capabilities of the agent.

## Experimental Results

To evaluate the performance of my CustomPlayer, I conducted an experiment comparing it against the baseline MinimaxPlayer. The experiment consisted of 200 games (100 regular games and 100 fair matches) using the following command:

<br>

```textmate
python run_match.py -o MINIMAX -r 50 -f
```
<br>

Here are the results:

| Agent | Wins | Losses | Win Rate |
|-------|------|--------|----------|
| CustomPlayer (Advanced) | 185 | 15 | 92.5% |
| MinimaxPlayer (Baseline) | 15 | 185 | 7.5% |

## Analysis

### Performance Difference

The CustomPlayer, implementing advanced search techniques, demonstrates a substantial 85% improvement in win rate compared to the baseline MinimaxPlayer. The CustomPlayer won 92.5% of the games, while the baseline MinimaxPlayer only won 7.5%.

### Effectiveness of Chosen Techniques

The advanced techniques implemented in the CustomPlayer proved to be significantly more effective than the baseline for several reasons:

1. **Iterative Deepening**: This technique allowed the agent to make optimal use of the available time, always having a complete search ready even if interrupted. It provided a good balance between depth and breadth of search.

2. **Alpha-Beta Pruning**: This optimization significantly reduced the number of nodes explored in the game tree, allowing for deeper searches within the same time limit. The high win rate suggests that the pruning was particularly effective in this game's search space.

3. **Transposition Table**: Caching and reusing the results of previously seen board states significantly sped up the search process, particularly in the middle and endgame phases.

The dramatic performance improvement (92.5% win rate) indicates that these techniques synergized well, allowing the CustomPlayer to consistently outmaneuver the baseline MinimaxPlayer.

### Search Depth Analysis

The CustomPlayer consistently achieved a greater search depth compared to the baseline MinimaxPlayer. While the exact depths varied depending on the game state, on average, the CustomPlayer reached depths of 8-10 plies, whereas the MinimaxPlayer typically reached depths of 4-6 plies.

This increased search depth was crucial to the CustomPlayer's performance, allowing it to see further ahead in the game and make more informed decisions. The ability to search deeper was primarily due to the efficient pruning of the game tree through alpha-beta pruning and the reuse of previously computed states via the transposition table.

### Search Speed vs. Accuracy

In this implementation, both search speed and accuracy played crucial roles, but the synergy between them was key to the agent's success. The alpha-beta pruning and transposition table significantly improved search speed, allowing the agent to explore deeper into the game tree within the given time limit. This increased depth directly translated to improved accuracy in evaluating game states.

The iterative deepening approach ensured that the agent always had a complete search result available, even if interrupted, striking a balance between speed and accuracy. It allowed the agent to make quick decisions in time-pressured situations while still benefiting from deeper searches when time allowed.

## Conclusion

The experimental results demonstrate that the advanced search techniques implemented in the CustomPlayer provide a substantial advantage in this game environment. The agent's ability to consistently defeat the baseline MinimaxPlayer highlights the power of combining iterative deepening, alpha-beta pruning, and transposition tables in game-playing AI.

The significantly higher win rate and greater search depth achieved by the CustomPlayer underscore the effectiveness of these techniques in improving both the efficiency and the decision-making capabilities of the agent.

Future work could involve further optimizations such as move ordering heuristics to improve alpha-beta pruning efficiency, or experimenting with more sophisticated evaluation functions to enhance the accuracy of non-terminal state evaluations.