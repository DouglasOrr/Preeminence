import preem as P
import random


class Agent(P.Agent):
    def __init__(self, min_attack=3):
        self.min_attack = min_attack

    def __repr__(self):
        return 'RandomAgent[min_attack={}]'.format(self.min_attack)

    def place(self, state):
        return random.choice(state.my_territories)

    def reinforce(self, state, count):
        return {random.choice(state.my_territories): count}

    def redeem(self, state):
        if 5 <= len(state.cards):
            sets = P.get_matching_sets(state.cards)
            return random.choice(sets)

    def act(self, state, earned_card):
        possible_attacks = [a for a in P.get_all_possible_attacks(state) if a.count >= self.min_attack]
        if possible_attacks:
            return random.choices(possible_attacks, weights=[a.count for a in possible_attacks])[0]
        possible_moves = P.get_all_possible_moves(state)
        if possible_moves:
            return random.choices(possible_moves, weights=[a.count for a in possible_moves])[0]
        return None
