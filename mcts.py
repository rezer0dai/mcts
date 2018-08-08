import math, random

from mcnode import *

class MCTS:
    def __init__(self, player_id, max_depth, prio):
        self.nodes = {}

        self.max_depth = max_depth
        self.prio = prio
        self.pid = player_id

    def scatter_root(self): # in case we want to reuse 'tree'
        root = self.nodes[None]

        for key in root.get_bandits(): # maybe make it with deep-2 to remove also other players state
            if key not in self.nodes:
                continue
            self.nodes.pop(key) # scattern unused actions

        key = self._load_node(root.state)
        self.nodes.pop(key) # scatter, reuse just subtree

    def prone_space(self, state): # wrapped monte carlo tree search
        key = self._load_node(state)
        root = self.nodes[key]
        self.nodes[None] = root

        self._monte_carlo_search(key)

        # get best move for now ( discard playing ration )
        _, action = self._exploit(root, 0)
        return action

    def _load_node(self, state):
        key = self.pid, state.board, state.player(), state.locs
        if key not in self.nodes:
            self.nodes[key] = MCNode(state)
        return key

    def _outcome(self, node):
        # here we outsource .state member, i dont like it, however to proper avoid needs :
        #  to proper rething state interface
        player_id = node.state.player() # who is playing now ?
        if node.state.utility(player_id) < 0:
            return 1
        if node.state.utility(player_id) > 0:
            return -1
        return 0 # dunno, too long game - dully noted

    def _explore(self, node):
        state, action = node.greedy()

        key = self._load_node(state)

        node.expand(key, action)
        return key, action

    def _exploit(self, node, c):
        if len(node.bandits) and (0 == c or not len(node.actions)):

            return max(
                node.bandits,
                key=lambda n_a: self.nodes[n_a[0]].ucb(node.plays, c),
                default=(None, None))

        return self._explore(node)

    def _step(self, root, depth):
        if root.state.terminal_test() or depth > self.max_depth:
            return self._outcome(root)

        next_key, _ = self._exploit(root, self.prio)
        if None == next_key:
            return self._outcome(root)

        return self._monte_carlo_search(next_key, depth + 1)

    def _monte_carlo_search(self, key, depth = 0):
        root = self.nodes[key]
        value = self._step(root, depth)
        return root.backprop(value)