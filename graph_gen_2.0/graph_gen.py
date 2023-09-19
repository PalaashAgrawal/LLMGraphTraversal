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
        random.seed(10000*random.random())

    def create_graph(self, level:int):

        f = self.get_level(level)
        graph = f(self.n)
        # self.n = len(graph.nodes) #sometimes, we add more nodes that n. Eg, level 3
        


        # Convert the graph to an adjacency matrix
        adj_matrix = nx.adjacency_matrix(graph, list(range(len(graph.nodes))))
        adj_array = np.array(adj_matrix.todense())
        
        

        node_order = list(range(len(graph.nodes)))
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
        
        if level ==3 or level ==4:
            def _get_level_3_4(n):
                assert n<=25, f'n cannot exceed 25. The adjacency matrix cannot be represented properly beyond that'

                f'use a random tree, then add random paths between random nodes in the shortest path of the original graph. '
                graph = nx.random_tree(n, seed = int(10000*random.random()))

                #relabel the nodes in order through a DFS traversal
                graph = self._get_ordered_graph(graph)

                shortest_path = nx.shortest_path(graph, source=0, target=n-1, weight=None, method='dijkstra')

                num_extra_paths = random.randint(1,2) #we want at max 2 additional paths in the graph
                for _ in range(num_extra_paths):
                    #decide what nodes to connect with an extra path
                    nodes_to_connect = random.sample(shortest_path, 2)
                    start_node, end_node = nodes_to_connect
                    #decide how many extra nodes should be there in the path that we are creating.
                    num_extra_nodes = random.randint(1, len(shortest_path))
                    
                    #total number of nodes should not exceed 26, otherwise we run out of letters 
                    if n+num_extra_nodes> 26: 
                        num_extra_nodes = 26-n
                        if num_extra_nodes <= 0: break
                        
                    n+=num_extra_nodes

                    #now create the new path
                    new_node = len(graph.nodes)
                    for i in range(num_extra_nodes):

                        graph.add_edge(start_node, new_node)
                        start_node = new_node
                        new_node+=1
                    graph.add_edge(start_node, end_node)
                
                if level ==4:
                    #add random weights 
                    for edge in graph.edges(): graph[edge[0]][edge[1]]['weight'] = random.randint(1, 5)

                
                
                
                return graph

            return _get_level_3_4
        
        if level ==5 or level==6:
            def _get_level_5_6(n):
                f'grid'
                m = int(math.sqrt(n))+1
                # dimensions = m,m+random.randint(0,2)
                dimensions = m,m
                
                graph = nx.grid_2d_graph(*dimensions)
                start, end = (0, 0), (dimensions[0]-1, dimensions[1]-1)


                #generate all shortest paths and simple paths
                random.seed(10000*random.random())
                shortest_path = random.choice(list(nx.all_shortest_paths(graph, start, end)))
                all_other_paths = list(nx.all_simple_paths(graph, start, end))

                #only keep simple paths that are longer than shortest path
                
                shortest_path_length = len(shortest_path)
                all_other_paths = [path for path in all_other_paths if len(path) > shortest_path_length]

                random.seed(10000*random.random())
                random.shuffle(all_other_paths)

                #randomly select 3-5 paths from all paths, and remove the edges for the rest
                kept_paths = random.sample(all_other_paths, random.randint(3,5))

                #convert lists into correct format
                kept_paths_edges = {(path[i], path[i+1]) for path in kept_paths for i in range(len(path) - 1)}
                shortest_path_edges = {(shortest_path[i], shortest_path[i+1]) for i in range(len(shortest_path) - 1)}

                #take the union
                all_paths = kept_paths_edges | shortest_path_edges

                #remove edges
                for edge in list(graph.edges):
                    if edge not in kept_paths_edges: graph.remove_edge(*edge)

                #remove any disconnected nodes. 
                graph = self._remove_disconnected_nodes(graph, start = (0,0))

                #renaming the nodes. 
                mapping = {node: k for k,node in enumerate(graph.nodes)}
                graph = nx.relabel_nodes(graph, mapping)

                #total number of nodes change after removing nodes. 
                self.n = len(graph.nodes)

                if level ==6:
                    #add random weights 
                    for edge in graph.edges(): graph[edge[0]][edge[1]]['weight'] = random.randint(1, 5)

                return graph 
            
                

            return _get_level_5_6
        

        if level ==7:
            
                





        




     
    
    def get_shortest_path(self, graph, source, target):
        shortest_path = nx.shortest_path(graph, source=source, target=target, weight='weight', method='dijkstra')
        ret = f''
        for node in shortest_path[:-1]: ret+=f'{node} -> '
        return ret+str(shortest_path[-1])
        

    def jumble_adj_matrix(self, matrix, is_directed = False):
        n = len(matrix)
        # Get a list of nodes and shuffle them to get a new sequence
        nodes = list(range(n))
        shuffled_nodes = self._shuffle_list(nodes)
        ordered_to_shuffled_mapping = {k:i for k,i in enumerate(shuffled_nodes)}


        
        new_matrix = np.zeros((n, n), dtype=int)
        for i in range(n): #row
            for j in range(n): #column
                new_matrix[ordered_to_shuffled_mapping[i]][ordered_to_shuffled_mapping[j]] = matrix[i][j]
                
        

            
        
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
    

    def _remove_disconnected_nodes(self,graph, start):
        connected_edges = set()
        for a,b in nx.edge_dfs(graph,start): 
            connected_edges.add(a)
            connected_edges.add(b)
        
        nodes_to_remove = []
        for node in graph.nodes:
            if node not in connected_edges: nodes_to_remove.append(node)
        
        graph.remove_nodes_from(nodes_to_remove)
        
        return graph



gen = create_level(n=10, is_jumbled=True)
out, sh = gen.create_graph(level = 5)
print(out)
print(sh)


# res = set()
# for i in range(50):
#     print(i)
#     gen = create_level(n=20, is_jumbled=False)
#     out, sh = gen.create_graph(level = 5)
#     res.add(sh)

# print(res, len(res))