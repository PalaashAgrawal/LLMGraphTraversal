import networkx as nx
import numpy as np


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
        mapping = {i: chr(65 + i) for i in range(self.n)}  # 65 is ASCII for 'A'
        graph = nx.relabel_nodes(graph, mapping)


        # Convert the graph to an adjacency matrix
        adj_matrix = nx.adjacency_matrix(graph)
        adj_array = np.array(adj_matrix.todense())

        out = self.convert_adjArray_to_str(adj_array, mapping)
        shortest_path = self.get_shortest_path(graph, mapping[0], mapping[self.n-1])


        


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

    
    def get_shortest_path(self, graph, source, target):
        shortest_path = nx.shortest_path(graph, source=source, target=target, weight=None, method='dijkstra')
        ret = f''
        for node in shortest_path[:-1]: ret+=f'{node} -> '
        return ret+shortest_path[-1] 
        







