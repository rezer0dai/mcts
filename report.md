# Monte Carlo Tree Search 

## Udacity AIND project 3 ( Adversial Search )

### Baselines
  * python run_match.py -r100 -f -o MINIMAX
    * used fair matches to avoid of exploitation of early advantage, because of first good opening state ~ on both sides
    * 100 rounds to get relatevily unbiased result of performance ( win or loose )

### Q&A
  * algorithm choosen for comparsion **MINIMAX**
    * 70~90 % of wins for my implementation of **MCTS**
  * mcts greedy plays whole games, wherareas minimax has access only for depth 3 and compare only liberties ~ which i dont think is good measurement of future win or loose
  * using mcts is in this game effective as we can given timeline ( 150 ms ) play several full end-games
    * if we need lots of move ( unsufficient to manage in one turn in most cases ) then i do believe minimax ( if good heuristic ~ better than liberties dif ) can get edge
  * also we are lucky that this game does not eat too much state + action space
    * if it will be opposite, then more feasible solution than hash-table/tree approach will be use function approximators ( deep neural nets )

### charts 
  * given following heuristic by my reviewer i did additional chart ( change applies to MiniMax section ) : 
    * heuristic here means, Opponent algorithm used original one ( followed by udacity implementation ), or reviewer one ( given code snapshot above )

```python
    def score(self, state):
        mid_w, mid_h = 11 // 2 + 1, 9 // 2 + 1
        center_location = (mid_w, mid_h)
        own_loc = state.locs[self.player_id]
        if center_location == own_loc:
            return self.heur(state) + 100
        return self.heur(state)

    def heur(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)
```

Opponent | MCTS | Heuristic
--- | --- | ---
Random | 97.5 ~ 100 % | original
Greedy| 97.5 ~ 100 % | original
MiniMax | 70 ~ 92 % | original
MiniMax| ~ 77.5 % | reviewer

  * here you can see that adding more appropriate ( in terms of reward ) heuristic for MiniMax actually helps against MCTS approach
    * clip MCTS success rate to 77.5% and oscilate closely ( without it was randomly - experienced - between range 70 .. 92 )
  * on the other side, once i applied same heuristic to my MCTS ( replacing _outcome {+1, -1, 0} with proposed shaped reward function from reviewer ) my MCTS score very poorly :
    * *Your agent won 17.5% of matches against Minimax Agent*
    * why is that ? i suppose for MCTS sparse reward function works better as still this is pretty complicated problem
      * sparse reward function will leave all heavy lifting to MC + UCB approach
      * while shaped reward function adding additional ( human crafted ) logic which may be biased towards our thinking rather then true function which represents given problem

### additional thoughts : 
  * replace max in UCB approach, with SumTree
    * while following UCB approach we will give more randomization per move
    * i implemented this during process but it leads to similiar or worse results
    * however i will need to retry with this final version
  * avoiding to scattering whole Game-Tree ( well in my case more hash-table / dictionary )
    * lost lot of computed information which should be valid ( well mathematical prooves probably will deny me out of blue :)
    * i think about preserving information in : 
      * whole game ( like AlphaGo, just scattering root and first/second order computations, and preserving all from new-state which is new root )
      * eterninty ~ well, why just preserve it trough one game, we can reuse between multiple games
        * then also perhaps scattering informations should be somehow bucketed, and per state we can have different saved game-tree to use