import networkx as nx
import random
import matplotlib.pyplot as plt

def add_extra_paths(graph, a, n):
    used_lengths = set()

    for x in range(a):
        if random.random() < 0.4:
          elements = random.sample([x for x in range(n)], k=2)
        else:
          elements = random.sample([x for x in range(n)], k = 1) + [9]
        elements.sort()  # Ensure the first node comes before the second

        while True:
            num_nodes = random.randint(1, n-5)
            if num_nodes not in used_lengths:
                used_lengths.add(num_nodes)
                break

        start_node = elements[0]

        for i in range(num_nodes):
            new_node = max(graph.nodes) + 1
            graph.add_edge(start_node, new_node)
            start_node = new_node

        if num_nodes > 0:
            graph.add_edge(start_node, elements[1])

    return graph

def add_weights(graph, min, max):
  for edge in graph.edges():
      graph[edge[0]][edge[1]]['weight'] = random.randint(min, max)
  return graph



size_of_graph = 10
repeats = 3

graph = nx.path_graph(size_of_graph)
graph = add_extra_paths(graph, repeats, size_of_graph)
min_weight, max_weight = 1, 3
graph = add_weights(graph, min_weight, max_weight)

pos = nx.spring_layout(graph)
nx.draw(graph, pos, with_labels=True)
labels = nx.get_edge_attributes(graph, 'weight')
nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)