import networkx as nx
import numpy as np

import random
import math

# For info on levels: refer to: 
# https://docs.google.com/document/d/1ckwAyQpFihcgMUOrmI4VS_5Lsi7AdCMl/edit?usp=sharing&ouid=108292991556701680365&rtpof=true&sd=true



class create_level():
    def __init__(self, n:int, is_jumbled:bool = False):
        f'''
        Input:
        n: number of nodes in the final graph
        is_jumbled: if True, nodes appear in a jumbled order. 

        returns: 
        adjacency matrix (str), with nodes represented as A,B,C...
        The shortest traversal path, represented as A->B->C.. #CHANGE THIS IS MODELS PERFORM WORSE WITH ARROWS. 

        '''

        self.n = n
        self.is_jumbled = is_jumbled

    def create_graph(self, level:int):

        f = self.get_level(level)
        graph = f(self.n)
        


        # Convert the graph to an adjacency matrix
        adj_matrix = nx.adjacency_matrix(graph, list(range(len(graph.nodes))))
        adj_array = np.array(adj_matrix.todense())
        
        node_order = list(range(self.n))
        if self.is_jumbled: adj_array, node_order = self.jumble_adj_matrix(adj_array)

        mapping = {k: chr(65 + i) for k,i in enumerate(node_order)}  # 65 is ASCII for 'A'
        graph = nx.relabel_nodes(graph, mapping)

        shortest_path = self.get_shortest_path(graph, mapping[0], mapping[self.n-1])
        
        out = self.convert_adjArray_to_str(adj_array, dict(sorted(mapping.items(), key=lambda x:x[1])))

        return out, shortest_path
    
    def convert_adjArray_to_str(self, adj_array, mapping):
        new_line = '\n'
        out = f''''''
        out+= f"   {' '.join(mapping.values())}"
        for row_label, row in zip(mapping.values(), adj_array):
            out+=f'''{new_line} {row_label} {' '.join(map(str, row))}'''

        return out

    def get_level(self, level):
        assert 1<=int(level)<=10

        if level==1:
            return  nx.path_graph
        
        if level==2:
            def _get_level2(n):
                f'use a random tree'
                graph = nx.random_tree(n, seed = int(10000*random.random()))
                
                #relabel the nodes in order through a DFS traversal
                graph = self._get_ordered_graph(graph)
                return graph

            return _get_level2

        


    
    
    def get_shortest_path(self, graph, source, target):
        shortest_path = nx.shortest_path(graph, source=source, target=target, weight=None, method='dijkstra')
        ret = f''
        for node in shortest_path[:-1]: ret+=f'{node} -> '
        return ret+shortest_path[-1] 
        

    def jumble_adj_matrix(self, matrix, is_directed = False):
        n = len(matrix)
        
        # Get a list of nodes and shuffle them to get a new sequence
        nodes = list(range(n))
        shuffled_nodes = self._shuffle_list(nodes)
        
        # Create a new adjacency matrix based on the shuffled sequence
        new_matrix = np.zeros((n, n), dtype=int)
        for i in range(n-1):
            new_matrix[shuffled_nodes[i]][shuffled_nodes[i+1]] = 1
            if not is_directed: new_matrix[shuffled_nodes[i+1]][shuffled_nodes[i]] = 1 #check logic later.

        
        return new_matrix, shuffled_nodes
    
    

    def _shuffle_list(self, arr):
        return sorted(list(range(len(arr))), key=lambda x: random.random())



    def _get_ordered_graph(self, graph, source = 0):
        f'Takes a networkx graph and relabels the nodes such that the node occurence is ordered. Logic -- dfs traversal iterates through the nodes in an ordered fashion'
        
        dfs_traversal = list(nx.dfs_edges(graph, source=source))
        mapping = {}
        for k,(_,b) in enumerate([(None,source)] + dfs_traversal): mapping[b] = k
        graph = nx.relabel_nodes(graph, mapping)
    
        return graph










gen = create_level(n=10)
out, sh = gen.create_graph(level = 2)
print(out)
print(sh)

# gen = create_level(6, is_jumbled=True)
# out, sh =gen.create_graph(1)
# print(out)
# print(sh)