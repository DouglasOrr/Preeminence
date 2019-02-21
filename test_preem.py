import pytest
import random
import collections
import os
import itertools as it
import unittest.mock as um
import numpy as np
import networkx as nx

import preem
import agents.random_agent as random_agent


# Utility & helper unit tests

def test_game_result():
    assert preem.GameResult([1, 2], [3, 0]).outright_winner() is None
    assert preem.GameResult([2], [3, 1, 0]).outright_winner() == 2


def test_get_matching_sets():  # implicitly tests is_matching_set
    assert list(preem.get_matching_sets([])) == []
    C = preem.Card
    assert list(preem.get_matching_sets([C(0, 10), C(1, 20), C(0, 30), C(1, 40)])) == []
    assert list(preem.get_matching_sets([C(1, 10), C(1, 20), C(1, 30)])) == \
        [(C(1, 10), C(1, 20), C(1, 30))]
    assert [tuple(c.symbol for c in set_)
            for set_ in preem.get_matching_sets([C(0, 10), C(1, 20), C(2, 30), C(2, 40), C(2, 50)])] == \
        [(0, 1, 2), (0, 1, 2), (0, 1, 2), (2, 2, 2)]


def test_value_of_set():
    assert [preem.value_of_set(n) for n in range(8)] == \
        [4, 6, 8, 10, 12, 15, 20, 25]


def test_count_reinforcements():
    assert preem.count_reinforcements(1) == 3
    assert preem.count_reinforcements(11) == 3
    assert preem.count_reinforcements(12) == 4
    assert preem.count_reinforcements(32) == 10


# Core data tests

def test_map_world_state_game_repr():
    map_ = preem.Map.load_file('maps/tiny3.json')
    game = preem.Game.start(map_, [random_agent.Agent()] * 2)
    first_placement = next(game)

    assert 'territories=3' in str(game.world.map)
    assert 'continents=1' in str(game.world.map)
    assert game.world.map._repr_svg_() is not None

    assert str(game.world.map) in str(game.world)
    assert 'players=3' in str(game.world)
    assert game.world._repr_svg_() is not None

    state = game.agents_and_states[0][1]
    state._add_cards([1, 2, 3, 4])  # don't actually need real cards here
    assert 'territories=1/3' in str(state)
    assert 'armies=1/3' in str(state)
    assert 'cards=4' in str(state)
    assert state._repr_svg_() is not None

    assert str(first_placement.state) in str(first_placement)
    assert first_placement._repr_svg_() is not None


# Core unit tests

def test_placement_phase():
    map_ = preem.Map.load_file('maps/tiny3.json')
    game = preem.Game.start(map_, [random_agent.Agent(), random_agent.Agent()])
    placement_events = list(it.takewhile(lambda e: e.method == 'place', game))
    # game should now be paused before evaluating the first reinforce()

    assert game.world.has_neutral
    n_armies = map_.initial_armies[2]
    assert len(placement_events) == 3 * n_armies - 3, 'initial armies minus assigned territories'
    assert len(set(id(e.agent) for e in placement_events[:3])) == 3, 'placement is round-robin'
    assert all(a == n_armies for a in game.world.armies)
    assert set(game.world.owners) == {0, 1, 2}


def test_placement_phase_asymmetric():
    random.seed(500)
    map_ = preem.Map.load_file('maps/tiny4.json')
    game = preem.Game.start(map_, [random_agent.Agent(), random_agent.Agent(), random_agent.Agent()])
    placement_events = list(it.takewhile(lambda e: e.method == 'place', game))
    # game should now be paused before evaluating the first reinforce()

    assert not game.world.has_neutral
    n_armies = map_.initial_armies[3]
    assert len(placement_events) == 3 * n_armies - 4, 'initial armies minus assigned territories'
    assert len(set(id(e.agent) for e in placement_events[:3])) == 3, 'placement is round-robin'

    armies_by_owner = collections.defaultdict(int)
    territories_by_owner = collections.defaultdict(int)
    for owner, armies in zip(game.world.owners, game.world.armies):
        armies_by_owner[owner] += armies
        territories_by_owner[owner] += 1
    assert all(n == n_armies for n in armies_by_owner.values())
    assert list(sorted(territories_by_owner.values())) == [1, 1, 2], 'someone random gets an extra territory'


def _make_world(map_, agents):
    world = preem.World(map_, [str(a) for a in agents], isinstance(agents[-1], preem._NeutralAgent))
    return world, [(agent, preem.PlayerState(world, n)) for n, agent in enumerate(agents)]


def test_placement_phase_error():
    random.seed(500)
    map_ = preem.Map.load_file('maps/tiny3.json')
    bad_agent = um.Mock(__str__=um.Mock(return_value='BadMockAgent'))
    agents = [bad_agent, random_agent.Agent(), random_agent.Agent()]

    world, agents_and_states = _make_world(map_, agents)
    bad_agent.place = um.Mock(return_value=3)  # out-of-bounds placement
    with pytest.raises(ValueError) as e:
        list(preem._placement_phase(world, agents_and_states, random))
    assert 'place' in str(e) and 'BadMockAgent' in str(e)
    assert '0..2' in str(e) and '3' in str(e)

    world, agents_and_states = _make_world(map_, agents)
    bad_agent.place = um.Mock(side_effect=lambda state: state.world.owners.index(1))  # enemy territory placement
    with pytest.raises(ValueError) as e:
        list(preem._placement_phase(world, agents_and_states, random))
    assert 'place' in str(e) and 'enemy' in str(e) and 'BadMockAgent' in str(e)
    assert '1' in str(e)


def test_placement_phase_too_many_players():
    random.seed(400)
    map_ = preem.Map.load_file('maps/tiny3.json')
    game = preem.Game.start(map_, [random_agent.Agent()] * 4)
    with pytest.raises(ValueError) as e:
        list(game)
    assert '4' in str(e)
    assert '3' in str(e)


def test_reinforce():
    map_ = preem.Map.load_file('maps/quad.json')
    tid = map_.territory_names.index
    C = preem.Card
    deck = um.Mock()

    # Set up the world
    world = preem.World(map_, ['p0', 'p1', 'p2'], has_neutral=False)
    for i in range(map_.n_territories):
        world.armies[i] = 1
        world.owners[i] = 2
    world.owners[tid('A4')] = 0
    world.owners[tid('B4')] = 1
    agent_0 = um.Mock()
    state_0 = preem.PlayerState(world, 0)

    # 1. Minimum territory reinforcements, no continents, no sets
    agent_0.reinforce = um.Mock(return_value={tid('A4'): 3})
    events = list(preem._reinforce(agent_0, state_0, deck))
    assert events == [preem.Event(agent_0, state_0, 'reinforce', dict(count=3), {tid('A4'): 3})]
    assert world.armies[tid('A4')] == 1 + 3
    assert world.sets_redeemed == 0

    # 2. Minimum territory, no continents, declaring a set
    cards = [C(0, tid('B1')), C(1, tid('B2')), C(2, tid('B3')), C(2, tid('B4'))]
    cards_to_redeem = [cards[0], cards[1], cards[3]]
    state_0.cards = cards.copy()
    world.n_cards[0] = len(cards)
    agent_0.redeem = um.Mock(return_value=cards_to_redeem.copy())
    agent_0.reinforce = um.Mock(return_value={tid('A4'): 7})  # 3 (territory) + 4 (set)
    events = list(preem._reinforce(agent_0, state_0, deck))
    assert events == [preem.Event(agent_0, state_0, 'redeem', {}, cards_to_redeem.copy()),
                      preem.Event(agent_0, state_0, 'reinforce', dict(count=7), {tid('A4'): 7})]
    assert world.armies[tid('A4')] == 4 + 7
    assert world.sets_redeemed == 1
    assert world.n_cards[0] == 1
    assert state_0.cards == [C(2, tid('B3'))]
    deck.redeem.assert_called_with(cards_to_redeem)

    # 3. Multiple territory, multiple continents, declaring a set
    agent_2 = um.Mock()
    cards = [C(2, tid('A2')), C(2, tid('A3')), C(2, tid('A4'))]
    state_2 = preem.PlayerState(world, 2, cards=cards)
    world.n_cards[2] = len(state_2.cards)
    agent_2.redeem = um.Mock(return_value=cards.copy())
    # Need to make 17 reinforcements: 6 (=18/3, territory) + 6 (set) + 5 (=2+3, continents)
    reinforcements = {tid('A3'): 11, tid('B5'): 6}
    agent_2.reinforce = um.Mock(return_value=reinforcements)
    events = list(preem._reinforce(agent_2, state_2, deck))
    assert events == [preem.Event(agent_2, state_2, 'redeem', {}, cards.copy()),
                      preem.Event(agent_2, state_2, 'reinforce', dict(count=17), reinforcements)]
    assert world.armies[tid('A4')] == 11, 'matches set but not owned, so unchanged'
    assert world.armies[tid('A2')] == 3, 'bonus armies'
    assert world.armies[tid('A3')] == 1 + 2 + 11, 'bonus armies & reinforcements'
    assert world.armies[tid('B5')] == 1 + 6, 'reinforcements'
    assert world.sets_redeemed == 2
    assert world.n_cards[2] == 0
    assert state_2.cards == []
    deck.redeem.assert_called_with(cards)


def test_reinforce_error():
    map_ = preem.Map.load_file('maps/tiny3.json')
    world = preem.World(map_, ['p0', 'p1', 'neutral'], has_neutral=True)
    world.armies = [1, 1, 1]
    world.owners = [0, 0, 1]

    agent_0 = um.Mock(__str__=um.Mock(return_value='BadMockAgent'))
    state_0 = preem.PlayerState(world, 0)

    # redeem
    C = preem.Card
    state_0.cards = [C(2, 0), C(2, 1), C(1, 2)]
    agent_0.redeem = um.Mock(return_value=[C(2, 0), C(2, 1), C(2, 2)])  # card not owned
    with pytest.raises(ValueError) as e:
        list(preem._reinforce(agent_0, state_0, um.Mock()))
    assert 'redeem' in str(e) and 'BadMockAgent' in str(e)

    agent_0.redeem = um.Mock(return_value=[C(2, 0), C(2, 1), C(1, 2)])  # not a matching set
    with pytest.raises(ValueError) as e:
        list(preem._reinforce(agent_0, state_0, um.Mock()))
    assert 'redeem' in str(e) and 'BadMockAgent' in str(e)

    state_0.cards = [C(2, 0), C(2, 1), C(1, 2), C(1, 0), C(0, 1), C(0, 2)]
    agent_0.redeem = um.Mock(return_value=None)  # fail to redeem with 6 cards
    with pytest.raises(ValueError) as e:
        list(preem._reinforce(agent_0, state_0, um.Mock()))
    assert 'redeem' in str(e) and 'BadMockAgent' in str(e)
    assert '6' in str(e) and '5' in str(e)

    # reinforce
    state_0.cards = []
    agent_0.redeem = um.Mock(return_value=None)

    agent_0.reinforce = um.Mock(return_value={2: 3})  # enemy territory
    with pytest.raises(ValueError) as e:
        list(preem._reinforce(agent_0, state_0, um.Mock()))
    assert 'reinforce' in str(e) and 'BadMockAgent' in str(e) and '2' in str(e)

    agent_0.reinforce = um.Mock(return_value={0: 2, 1: 2})  # selected the wrong number of reinforcements
    with pytest.raises(ValueError) as e:
        list(preem._reinforce(agent_0, state_0, um.Mock()))
    assert 'reinforce' in str(e) and 'BadMockAgent' in str(e)
    assert'4' in str(e) and '3' in str(e)


def test_attack_and_move():
    map_ = preem.Map.load_file('maps/tiny3.json')
    world = preem.World(map_, ['p0', 'p1', 'neutral'], has_neutral=True)
    world.owners = [0, 1, 2]
    world.armies = [5, 2, 1]

    agents = [um.Mock(), um.Mock(), None]
    states = [preem.PlayerState(world, 0),
              preem.PlayerState(world, 1, cards=[preem.Card(2, 2)]),
              preem.PlayerState(world, 2)]
    agents_and_states = list(zip(agents, states))
    deck = um.Mock()
    rand = um.Mock()

    # 1. Don't take any action - no card earnt
    agents[0].act = um.Mock(return_value=None)
    events = list(preem._attack_and_move(agents[0], states[0], deck, agents_and_states, rand))
    assert events == [preem.Event(agents[0], states[0], 'act', dict(earned_card=False), None)]
    assert len(states[0].cards) == 0, 'no cards earnt'

    # 2. Attack (knock out Neutral) then Move
    agents[0].act = um.Mock(side_effect=[preem.Attack(0, 2, 3), preem.Move(2, 0, 1)])
    rand.choices = um.Mock(return_value=[(0, 1)])
    events = list(preem._attack_and_move(agents[0], states[0], deck, agents_and_states, rand))
    assert events == [preem.Event(agents[0], states[0], 'act', dict(earned_card=False), preem.Attack(0, 2, 3)),
                      preem.Event(agents[0], states[0], 'act', dict(earned_card=True), preem.Move(2, 0, 1))]
    assert len(states[0].cards) == 1, 'earnt a card'
    assert world.owners == [0, 1, 0]
    assert world.armies == [3, 2, 2]
    # dice rolls
    alternatives, probabilities = rand.choices.call_args[0]
    assert alternatives == ((0, 1), (1, 0))
    np.testing.assert_almost_equal(probabilities, [855/1296, 441/1296])

    # 3. Attack (knock out other player, claim their cards), win
    agents[0].act = um.Mock(return_value=preem.Attack(0, 1, 2))
    rand.choices = um.Mock(return_value=[(0, 2)])
    with pytest.raises(preem._GameOverException):
        list(preem._attack_and_move(agents[0], states[0], deck, agents_and_states, rand))
    assert world.owners == [0, 0, 0]
    assert world.armies == [1, 2, 2]
    # dice rolls
    alternatives, probabilities = rand.choices.call_args[0]
    assert alternatives == ((0, 2), (1, 1), (2, 0))
    np.testing.assert_almost_equal(probabilities, [295/1296, 420/1296, 581/1296])
    # cards
    assert len(states[0].cards) == 2, 'conquered a card (haven\'t earnt one as the game ended too soon)'
    assert preem.Card(2, 2) in states[0].cards
    assert len(states[1].cards) == 0, 'conquered = lost card'
    assert world.n_cards == [2, 0, 0]


def test_attack_and_move_error():
    map_ = preem.Map.load_file('maps/tiny4.json')
    # disconnect (0, 2), so that there can be a "not connected" error
    map_.edges[0].remove(2)
    map_.edges[2].remove(0)
    world = preem.World(map_, ['p0', 'p1', 'neutral'], has_neutral=True)
    world.armies = [5, 5, 3, 5]
    world.owners = [0, 0, 1, 1]

    agents = [um.Mock(), um.Mock(), None]
    states = [preem.PlayerState(world, 0), preem.PlayerState(world, 1), preem.PlayerState(world, 2)]
    agents_and_states = list(zip(agents, states))
    rand = um.Mock()

    for action in [preem.Attack(0, 1, 2),  # attacking from enemy
                   preem.Attack(2, 3, 2),  # attacking to friendly
                   preem.Attack(2, 1, 3),  # attacking with too many units
                   preem.Attack(2, 0, 2),  # attacking to disconnected territory
                   preem.Move(2, 1, 2),    # moving to enemy
                   preem.Move(1, 2, 2),    # moving from enemy
                   preem.Move(2, 3, 3)]:   # moving too many units
        agents[1].act = um.Mock(return_value=action)
        with pytest.raises(ValueError) as e:
            list(preem._attack_and_move(agents[1], states[1], um.Mock(), agents_and_states, rand))
        assert str(action) in str(e)

    # sanity check that a correct attack & move are indeed possible
    agents[1].act = um.Mock(side_effect=[preem.Attack(2, 1, 2), preem.Move(3, 2, 4)])
    rand.choices = um.Mock(return_value=[(2, 0)])  # attacker loses
    assert len(list(preem._attack_and_move(agents[1], states[1], um.Mock(), agents_and_states, rand))) == 2
    assert len(states[1].cards) == 0, "didn't earn a card"
    assert world.armies == [5, 5, 5, 1]
    assert world.owners == [0, 0, 1, 1]


# Functional tests


MAPS = [
    'tiny3',
    'tiny4',
    'mini',
    'quad',
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

    # SVG repr
    assert map_._repr_svg_()

    # Graph
    g = map_.to_graph
    assert len(g) == map_.n_territories
    assert nx.algorithms.is_connected(g), 'can run NetworkX algorithms'


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


def test_play_game():
    random.seed(345)
    map_ = preem.Map.load_file('maps/tiny4.json')
    agents = [ConsistencyCheckingAgent(random_agent.Agent()),
              ConsistencyCheckingAgent(random_agent.Agent())]
    game = preem.Game.start(map_, agents)
    for n, event in enumerate(game):
        # log should already contain this latest event
        assert game.world.event_log[-1] == event._replace(agent=str(event.agent))
        assert len(game.world.event_log) == n + 1
    assert game.world._repr_svg_()
    assert game.world.event_log[-1]._repr_svg_()
    # game should be over
    assert len(game.result.winners) == 1 or game.world.turn == map_.max_turns


def test_watch_game():
    random.seed(987)
    map_ = preem.Map.load_file('maps/tiny3.json')
    agents = [ConsistencyCheckingAgent(random_agent.Agent()),
              ConsistencyCheckingAgent(random_agent.Agent())]
    name = 'test_watch_game_tmp.mp4'
    try:
        preem.Game.watch(map_, agents, name)
        assert os.path.isfile(name)
    finally:
        if os.path.exists(name):
            os.remove(name)


@pytest.mark.parametrize('map_name', MAPS)
def test_fuzz(map_name):
    random.seed(100)
    map_ = preem.Map.load_file('maps/{}.json'.format(map_name))
    rand_agent = ConsistencyCheckingAgent(random_agent.Agent())
    for n_players in range(2, map_.max_players):
        agents = [rand_agent] * n_players
        ntrials = 10 if map_name == 'classic' else 1000
        results = [preem.Game.play(map_, agents) for _ in range(ntrials)]
        for result in results:
            assert set(result.winners) | set(result.eliminated) == set(range(n_players))
            assert not (set(result.winners) & set(result.eliminated))
        if map_name in {'mini', 'tiny3', 'tiny4'}:
            winners = dict(collections.Counter([r.outright_winner() for r in results]))
            # small maps, aggressive agents => there should normally be a winner
            n_draws = winners.pop(None, 0)
            assert n_draws < 0.1 * ntrials
            # fairness means the wins should be distributed evenly
            # - this test could fail - if so, try increasing ntrials
            p_win = 1 / n_players
            norm_approx_std = (ntrials * p_win * (1 - p_win)) ** 0.5
            assert max(winners.values()) - min(winners.values()) < 3 * norm_approx_std
