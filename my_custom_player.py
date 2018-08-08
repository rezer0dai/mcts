###########################################################
# benchmarked against ( borrowo 1/match.sqrt(2) + ucb(0) idea :
#   https://github.com/bobtal/artificial-intelligence/blob/master/Projects/3_Adversarial Search/my_custom_player.py
###########################################################

import os, pickle

from sample_players import DataPlayer

from mcts import *

class CustomPlayer(DataPlayer):
    def get_action(self, state):
        self.queue.put(random.choice(state.actions()))
        if state.ply_count < 2:
            return
        self.play(state)

    def __init__(self, player_id):
        super(CustomPlayer, self).__init__(player_id)
        self.keep = False
        self.mcts = MCTS(player_id, 40, 1 / math.sqrt(2))

        #  self.keep = True
        #  self.mcts = MCTS(player_id, 20, 1 / math.sqrt(2))

    def play(self, state):
        self._load_timeoverkill_udacityfail()
        while True:
            a = self.mcts.prone_space(state)
            self.queue.put(a)
            self._save_timeoverkill_udacityfail()

    def _save_timeoverkill_udacityfail(self):
        if not self.keep:
            return
        with open("mcts.pickle", "wb") as dtb:
            pickle.dump(self.mcts, dtb)

    def _load_timeoverkill_udacityfail(self):
        if not self.keep:
            return self.mcts
        if not os.path.exists("mcts.pickle"):
            return self.mcts

        self.mcts = None
        while None == self.mcts:
            with open("mcts.pickle", "rb") as dtb:
                self.mcts = pickle.load(dtb)

        self.mcts.scatter_root()
        return self.mcts
