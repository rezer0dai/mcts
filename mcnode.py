import math, random

class MCNode():
    def __init__(self, state):
        self.state = state
        self.actions = state.actions()

        self.bandits = []
        self.plays = 0
        self.value = 0

    def ucb(self, n, c):
        self_visit_n = self.plays if self.plays else 1
        total_visit_n = n if n != 0 else 1
        return (self.value / self_visit_n) + (c * math.sqrt(2 * math.log(total_visit_n) / self_visit_n))

    def greedy(self):
        action = random.choice(self.actions)
        state = self.state.result(action)
        return state, action

    def expand(self, key, action):
        if [key, action] in self.bandits:
            return
        self.bandits.append([key, action])

        if action not in self.actions:
            return
        self.actions.remove(action)

    def backprop(self, value):
        self.plays += 1
        self.value += value
        return -value

    def get_bandits(self):
        for b, a in self.bandits:
            yield b