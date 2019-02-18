import preem
import random


class Agent(preem.Agent):
    def __init__(self, min_to_attack=3):
        self.min_to_attack = min_to_attack

    def __repr__(self):
        return 'RandomAgent[min_to_attack={}]'.format(self.min_to_attack)

    def place(self, state):
        return random.choice(state.my_territories)

    def reinforce(self, state, count):
        return {random.choice(state.my_territories): count}

    def redeem(self, state):
        if 5 <= len(state.cards):
            sets = list(preem.get_matching_sets(state.cards))
            return random.choice(sets)

    def act(self, state, earned_card):
        territory = max(state.my_territories, key=lambda t: state.world.armies[t])
        armies = state.world.armies[territory]
        if self.min_to_attack <= armies:
            enemy_neighbours = [t for t in state.map.edges[territory]
                                if state.world.owners[t] != state.player_index]
            if enemy_neighbours:
                return preem.Attack(from_=territory, to=random.choice(enemy_neighbours), count=armies - 1)
            return preem.Move(from_=territory,
                              to=random.choice(list(state.map.edges[territory])),
                              count=armies - 1)
