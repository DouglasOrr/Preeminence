import random
import networkx as nx
import preem as P
from agents import random_agent


map_ = P.Map.load('maps/classic.json')
map_.max_turns = 10  # otherwise it is far too long with just these silly random agents!
agents = [random_agent.Agent()] * 3
seed = 42

# eg_classic.svg
random.seed(seed)
game = P.Game.start(map_, agents)
event = next(game)
with open('docs/eg_classic.svg', 'w') as f:
    f.write(game.world._repr_svg_())

# eg_classic.mp4
random.seed(seed)
P.Game.watch(map_, agents, 'docs/eg_classic.mp4')

# agent_flow.svg
g = nx.DiGraph()
g.graph.update(rankdir='LR', pad=.2)
g.add_nodes_from(['place', 'redeem', 'reinforce', 'act'])
g.add_edges_from([('place', 'place', dict(label='initial armies')),
                  ('place', 'redeem', dict(label='first turn')),
                  ('redeem', 'reinforce'),
                  ('reinforce', 'act'),
                  ('act', 'act', dict(label='attacks')),
                  ('act', 'redeem', dict(label='end of turn'))])
with open('docs/agent_flow.svg', 'w') as f:
    f.write(nx.nx_agraph.to_agraph(g).draw(prog='dot', format='svg').decode('utf8'))
