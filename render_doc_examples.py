import random
import os
import sys
import networkx as nx
import preem as P
from agents import random_agent


map_ = P.Map.load('maps/classic.json')
map_.max_turns = 10  # otherwise it is far too long with just these silly random agents!
agents = [random_agent.Agent()] * 3


def eg_classic_svg(name):
    random.seed(42)
    game = P.Game.start(map_, agents)
    next(game)
    with open(name, 'w') as f:
        f.write(game.world._repr_svg_())


def eg_classic_mp4(name):
    random.seed(42)
    P.Game.watch(map_, agents, name)


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


if __name__ == '__main__':
    root = 'docs/img'
    if not os.path.isdir(root):
        os.makedirs(root)
    for name, create in [('eg_classic.svg', eg_classic_svg),
                         ('eg_classic.mp4', eg_classic_mp4),
                         ('agent_flow.svg', agent_flow_svg)]:
        path = os.path.join(root, name)
        if '--force' in sys.argv or not os.path.isfile(path):
            sys.stderr.write('[render_doc_examples.py] Rendering {}\n'.format(path))
            create(path)
