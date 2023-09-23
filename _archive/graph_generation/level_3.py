import networkx as nx
import random
import matplotlib.pyplot as plt

def add_paths(graph, a, n):
    used_lengths = set()

    for x in range(a):
        #weight the connection toward connecting to end node
        if random.random() < 0.4:
          elements = random.sample([x for x in range(n)], k=2)
        else:
          elements = random.sample([x for x in range(n)], k = 1) + [9]
        elements.sort()  

        #don't repeat connection lengths(higher chance of multiple shortest)
        while True:
            num_nodes = random.randint(1, n-5)
            if num_nodes not in used_lengths:
                used_lengths.add(num_nodes)
                break

        start_node = elements[0]
        # add the connection
        for i in range(num_nodes):
            new_node = max(graph.nodes) + 1
            graph.add_edge(start_node, new_node)
            start_node = new_node

        # connect back to the end element
        if num_nodes > 0:
            graph.add_edge(start_node, elements[1])

    return graph

size_of_graph = 10
repeats = 3

graph = nx.path_graph(size_of_graph)
graph = add_paths(graph, repeats, size_of_graph)

nx.draw(graph, with_labels=True)