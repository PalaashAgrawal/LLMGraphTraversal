import networkx as nx
n=10
graph = nx.random_tree(n)
nx.draw(graph, with_labels=True)