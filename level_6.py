import networkx as nx
import random
import matplotlib.pyplot as plt

n = 4
graph = nx.grid_2d_graph(n,n)
start, end = (0, 0), (n-1,n-1)

#generate all shortest paths and simple paths
all_shortest_paths = list(nx.all_shortest_paths(graph, start, end))[0]
all_other_paths = list(nx.all_simple_paths(graph, start, end))

#only keep simple paths that are longer than shortest path
shortest_path_length = len(all_shortest_paths)
all_other_paths = [path for path in all_other_paths if len(path) > shortest_path_length]

#select 4 random simple paths
kept_paths = random.sample(all_other_paths, min(4, len(all_other_paths)))

#convert lists into correct format
kept_paths_edges = {(path[i], path[i+1]) for path in kept_paths for i in range(len(path) - 1)}
shortest_path_edges = {(all_shortest_paths[i], all_shortest_paths[i+1]) for i in range(len(all_shortest_paths) - 1)}

all_paths = kept_paths_edges | shortest_path_edges

#remove edges
for edge in list(graph.edges):
    if edge not in kept_paths_edges:
        graph.remove_edge(*edge)
        if not nx.is_connected(graph):
            graph.add_edge(*edge)

# Assign weights
for edge in shortest_path_edges:
    if graph.has_edge(edge[0], edge[1]):
        graph[edge[0]][edge[1]]['weight'] = random.randint(1,3)

for edge in kept_paths_edges - shortest_path_edges:
    if graph.has_edge(edge[0], edge[1]):
        graph[edge[0]][edge[1]]['weight'] = random.randint(3,6)

#visualize it with edge labels showing weights
pos = nx.spring_layout(graph)
nx.draw(graph, pos, with_labels=True)
labels = nx.get_edge_attributes(graph, 'weight')
nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)