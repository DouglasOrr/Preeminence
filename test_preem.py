import pytest
import random
import collections

import preem
import agents.random_agent as random_agent


# Unit tests

def test_game_result():
    assert preem.GameResult([1, 2], [3, 0]).outright_winner() is None
    assert preem.GameResult([2], [3, 1, 0]).outright_winner() == 2


# Functional tests


MAPS = [
    'tiny3',
    'tiny4',
    'mini',
    'classic',
]


@pytest.mark.parametrize('map_name', MAPS)
def test_maps(map_name):
    map_ = preem.Map.load_file('maps/{}.json'.format(map_name))
    assert map_.name == map_name
    assert map_name in str(map_)

    # Gameplay
    assert 0 < map_.max_turns
    assert 2 <= map_.max_players
    for n_players, initial_armies in map_.initial_armies.items():
        assert max(3, n_players) * initial_armies >= map_.n_territories

    # Territories
    assert map_.n_territories \
        == len(map_.territory_names) \
        == len(map_.continents) \
        == len(map_.edges)
    for territory, edges in enumerate(map_.edges):
        assert isinstance(edges, set)
        assert territory not in edges, 'cannot connect to yourself'
        assert 1 <= len(edges), 'every territory must be connected to at least one other'
        assert all(territory in map_.edges[e] for e in edges), 'all edges are symmetric'

    # Continents
    assert map_.n_continents \
        == len(map_.continent_names) \
        == len(map_.continent_values)
    assert set(map_.continents) == set(range(map_.n_continents)), 'each continent must match a territory'


class ConsistencyCheckingAgent(preem.Agent):
    """Wrap preem.Agent to check consistency of the world every time it is called."""
    def __init__(self, agent):
        self.agent = agent

    def _verify(self, state):
        world = state.world
        map_ = state.map

        # PlayerState
        assert 0 <= state.player_index < state.world.n_players
        assert 0 < len(state.my_territories), 'agents with no territories should be eliminated'
        assert len(state.cards) == world.n_cards[state.player_index], 'consistent card count'

        # World
        assert world.n_players \
            == len(world.player_names) \
            == len(world.n_cards)
        assert map_.n_territories \
            == len(world.owners) \
            == len(world.armies)

        assert 0 <= world.turn < map_.max_turns
        assert all(0 < armies for armies in world.armies), 'no empty territories'
        if world.has_neutral:
            assert world.n_players == 3

        active_players = set(world.owners)
        # remove neutral from active_players
        if world.has_neutral and (world.n_players - 1) in active_players:
            active_players.remove(world.n_players - 1)
        assert 2 <= len(active_players), 'if there is only one active player, the game is over'
        assert all(player not in active_players for player in world.eliminated_players)
        assert len(active_players) + len(world.eliminated_players) + world.has_neutral == world.n_players, \
            'all players either active or eliminated'

    def place(self, state):
        self._verify(state)
        return self.agent.place(state)

    def redeem(self, state):
        self._verify(state)
        return self.agent.redeem(state)

    def reinforce(self, state, count):
        self._verify(state)
        return self.agent.reinforce(state, count)

    def act(self, state, earned_card):
        self._verify(state)
        return self.agent.act(state, earned_card)


@pytest.mark.parametrize('map_name', MAPS)
def test_fuzz_fair(map_name):
    rand = random.Random(100)
    map_ = preem.Map.load_file('maps/{}.json'.format(map_name))
    rand_agent = ConsistencyCheckingAgent(random_agent.Agent())
    for n_players in range(2, map_.max_players):
        agents = [rand_agent] * n_players
        ntrials = 10 if map_name == 'classic' else 1000
        results = [preem.Game.play(map_, agents, rand=rand) for _ in range(ntrials)]
        for result in results:
            assert set(result.winners) | set(result.eliminated) == set(range(n_players))
            assert not (set(result.winners) & set(result.eliminated))
        if map_name in {'mini', 'tiny3', 'tiny4'}:
            winners = dict(collections.Counter([r.outright_winner() for r in results]))
            # small maps, aggressive agents => there should be a winner
            n_draws = winners.pop(None, 0)
            assert n_draws < 0.1 * ntrials
            # fairness means the wins should be distributed evenly
            # - this test could fail, if so, try increasing ntrials
            norm_approx_std = (ntrials / n_players) ** 0.5
            assert max(winners.values()) - min(winners.values()) < 3 * norm_approx_std
