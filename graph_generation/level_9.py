import random
import networkx as nx
n = 10
deg_sequence = [2*(random.randint(1,3)) for x in range(n)]
graph = nx.random_degree_sequence_graph(deg_sequence, seed=None, tries=10)
nx.draw(graph, with_labels=True)