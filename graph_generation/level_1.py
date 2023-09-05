import networkx as nx
n=6
graph = nx.path_graph(n)
nx.draw(graph, with_labels=True)