import random
import os
import sys
import shutil
import networkx as nx
import preem as P
from agents import random_agent


map_classic = P.Map.load('maps/classic.json')
map_classic.max_turns = 10  # otherwise it is far too long with just these silly random agents!
classic_agents = [random_agent.Agent()] * 3


def eg_classic_svg(name):
    random.seed(42)
    game = P.Game.start(map_classic, classic_agents)
    next(game)
    with open(name, 'w') as f:
        f.write(game.world._repr_svg_())


def eg_classic_mp4(name):
    random.seed(42)
    P.Game.watch(map_classic, classic_agents, name, dpi=144)


def agent_flow_svg(name):
    g = nx.DiGraph()
    g.graph.update(rankdir='LR', pad=.2)
    g.add_nodes_from(['place', 'redeem', 'reinforce', 'act'])
    g.add_edges_from([('place', 'place', dict(label='initial armies')),
                      ('place', 'redeem', dict(label='first turn')),
                      ('redeem', 'reinforce'),
                      ('reinforce', 'act'),
                      ('act', 'act', dict(label='attacks')),
                      ('act', 'redeem', dict(label='end of turn'))])
    with open(name, 'w') as f:
        f.write(nx.nx_agraph.to_agraph(g).draw(prog='dot', format='svg').decode('utf8'))


def mini(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)

    map_ = P.Map.load('maps/mini.json')
    agents = [random_agent.Agent(min_attack=1)] * 3
    seed = 1000

    random.seed(seed)
    game = P.Game.start(map_, agents)
    with open('{}/start.svg'.format(path), 'w') as f:
        f.write(next(game).state.world._repr_svg_())
    with open('{}/end.svg'.format(path), 'w') as f:
        f.write(list(game)[-1].state.world._repr_svg_())

    random.seed(seed)
    for n, event in enumerate(P.Game.start(map_, agents)):
        with open('{}/step_{}.svg'.format(path, n), 'w') as f:
            f.write(event._repr_svg_())

    random.seed(seed)
    P.Game.watch(map_, agents, '{}/game.mp4'.format(path), dpi=144)


def maps(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)
    for file in os.listdir('maps'):
        map_name = os.path.splitext(file)[0]
        map_ = P.Map.load('maps/{}'.format(file))
        assert map_.name == map_name
        with open('{}/{}.svg'.format(path, map_name), 'w') as f:
            f.write(map_._repr_svg_())


if __name__ == '__main__':
    root = 'docs/img'
    if not os.path.isdir(root):
        os.makedirs(root)
    for name, create in [('eg_classic.svg', eg_classic_svg),
                         ('eg_classic.mp4', eg_classic_mp4),
                         ('agent_flow.svg', agent_flow_svg),
                         ('mini', mini),
                         ('maps', maps)]:
        path = os.path.join(root, name)
        if '--force' in sys.argv or not os.path.exists(path):
            sys.stderr.write('[render_doc_examples.py] Rendering {}\n'.format(path))
            create(path)
