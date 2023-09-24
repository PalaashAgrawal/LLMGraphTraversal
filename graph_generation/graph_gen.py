import networkx as nx
import numpy as np

import random
import math

# For info on levels: refer to: 
# https://docs.google.com/document/d/1ckwAyQpFihcgMUOrmI4VS_5Lsi7AdCMl/edit?usp=sharing&ouid=108292991556701680365&rtpof=true&sd=true


def get_graph_and_solution(n, level, is_jumbled = False):
    f'given number of nodes, which level of graph to create and if the adjacency matrix has to be jumbled, get a valid adjacency matrix and solution'

    assert 1<=level<=10, f'invalid level'
    out, sh = f'', f''

    # if 1<=level<=7 or level ==9:
    #     gen = create_level(n=n, is_jumbled=is_jumbled)
    #     out, sh, nodes_to_traverse, nodes_in_graph = gen.create_graph(level = level)
    
    if level==8:
        while not sh.startswith('No'): #this is only for level 8

            gen = create_level(n=n, is_jumbled=is_jumbled)
            out, sh, nodes_to_traverse, nodes_in_graph = gen.create_graph(level = level)

    else: 
    
    # if level==10:
        gen = create_level(n=n, is_jumbled=is_jumbled)
        out, sh, nodes_to_traverse, nodes_in_graph = gen.create_graph(level = level)
        
        # return out, sh, nodes
    return out, sh, nodes_to_traverse, nodes_in_graph








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

        f'''
        returns
        out - adjacency matrix in the form of a string, where the nodes are labelled A,B,C and so on. 

        solution - graph traversal solution. Each graph has only one shortest solution possible. Format = A -> B -> C and so on. 
        For levels 9 (is_eulerian problem), we return a specific string indicating if the graph is eulerian
        For level 10, we provide 2 paths together corresponding to the 2 parts of the graph traversal. 

        nodes_to_traverse - the starting and the ending node (represented by a letter of the alphabet) between which traversal happens. 
        For level 9, we only provide the starting node (starting from which, if a eulerian path is possible or not)
        For level 10, we provide 3 nodes, corresponding to the nodes that need to be covered in the graph traversal

        max(node_order) - represents how many nodes there are in the graph (minus 1 because node order is 0-ordered). 
        Hence the adjacency graph goes from A to chr(65 + max(node_order))
        '''
        
        if 1<=level<=8:
            self.level = level
            
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
        
            
            out = self.convert_adjArray_to_str(adj_array, dict(sorted(mapping.items(), key=lambda x:x[1])))
            
            
            try:
                

                shortest_path = self.get_shortest_path(graph, mapping[0], mapping[self.n-1])
                graph.clear()

                return out, shortest_path, (mapping[0], mapping[self.n-1]), max(node_order)
            
            
            except nx.exception.NetworkXNoPath as e: #in the case of level 8
                return out, f'No possible path from {mapping[0]} to {mapping[self.n-1]}', (mapping[0], mapping[self.n-1]), max(node_order)
        
        elif level==9:
            self.level = level
            
            f = self.get_level(level)
            graph, is_eulerian, starting_node = f(self.n)
            
            # for cases when the graph is not eulerian, we just ask the model to start traversal from node 0. 
            starting_node = 0 if starting_node is None else starting_node

            # Convert the graph to an adjacency matrix
            adj_matrix = nx.adjacency_matrix(graph, list(range(len(graph.nodes))))
            adj_array = np.array(adj_matrix.todense())
            
            

            node_order = list(range(len(graph.nodes)))
            if self.is_jumbled: adj_array, node_order = self.jumble_adj_matrix(adj_array)

            mapping = {k: chr(65 + i) for k,i in enumerate(node_order)}  # 65 is ASCII for 'A'
            graph = nx.relabel_nodes(graph, mapping)
        
            
            out = self.convert_adjArray_to_str(adj_array, dict(sorted(mapping.items(), key=lambda x:x[1]))) 

            solution = f'This is a valid eulerian graph' if is_eulerian else f'This is not a valid eulerian graph'
            return out, solution, starting_node,  max(node_order)
        
        elif level==10:
            f'for level 10, we will just use a random tree with weights, ie level 4'
            self.level = 4

            f = self.get_level(self.level)
            graph = f(self.n)

            # Convert the graph to an adjacency matrix
            adj_matrix = nx.adjacency_matrix(graph, list(range(len(graph.nodes))))
            adj_array = np.array(adj_matrix.todense())
            
            node_order = list(range(len(graph.nodes)))
            if self.is_jumbled: adj_array, node_order = self.jumble_adj_matrix(adj_array)

            mapping = {k: chr(65 + i) for k,i in enumerate(node_order)}  # 65 is ASCII for 'A'

            graph = nx.relabel_nodes(graph, mapping)
        
            out = self.convert_adjArray_to_str(adj_array, dict(sorted(mapping.items(), key=lambda x:x[1]))) 

            # node_order = sorted([i for _,i in mapping.items()])
            node_order_ = [i for _,i in mapping.items()]
            random_node = random.choice(node_order_[1:-1])
            

            shortest_path = f'''Path from {node_order_[0]} to {random_node}: {self.get_shortest_path(graph, node_order_[0], random_node)}\nPath from {random_node} to {node_order_[-1]}: {self.get_shortest_path(graph, random_node, node_order_[-1])}'''

            return out, shortest_path, (node_order[0], random_node, node_order[-1]),  max(node_order)
            
            



    def get_level(self, level):
    
    
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
                    nodes_to_connect = random.choices(shortest_path, k=2)
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
                m = int(math.sqrt(n))+1 #+1 because otherwise for n =10, we get a 3x3 grids, where the total amount of permutations are not enough. 
                # dimensions = m,m+random.randint(0,2)
                dimensions = m,m
                
                graph = nx.grid_2d_graph(*dimensions)
                start, end = (0, 0), (dimensions[0]-1, dimensions[1]-1)


                #generate all shortest paths and simple paths
                random.seed(10000*random.random())
                shortest_path = random.choice(list(nx.all_shortest_paths(graph, start, end)))
                all_paths = list(nx.all_simple_paths(graph, start, end))

                #only keep simple paths that are longer than shortest path
                
                shortest_path_length = len(shortest_path)
                all_other_paths = [path for path in all_paths if len(path) > shortest_path_length]

                random.seed(10000*random.random())
                random.shuffle(all_other_paths)

                #randomly select 3-5 paths from all paths, and remove the edges for the rest
                kept_paths = random.choices(all_other_paths, k=random.randint(3,5))

                #convert lists into correct format
                kept_paths_edges = {(path[i], path[i+1]) for path in kept_paths for i in range(len(path) - 1)}
                shortest_path_edges = {(shortest_path[i], shortest_path[i+1]) for i in range(len(shortest_path) - 1)}

                #take the union
                all_paths = list(kept_paths_edges | shortest_path_edges)

                #remove edges
                for edge in list(graph.edges):
                    # if edge not in kept_paths_edges: graph.remove_edge(*edge)
                    if edge not in all_paths: graph.remove_edge(*edge)

                #remove any disconnected nodes. 
                graph = self._remove_disconnected_nodes(graph, start = (0,0))

                #renaming the nodes. 
                nodes = sorted([node for node in graph.nodes], key= lambda x: m*x[0] + x[1])
                mapping = {node: k for k,node in enumerate(nodes)}
                graph = nx.relabel_nodes(graph, mapping)

                #total number of nodes change after removing nodes. 
                self.n = len(graph.nodes)

                if level ==6:
                    #add random weights 
                    for edge in graph.edges(): graph[edge[0]][edge[1]]['weight'] = random.randint(1, 5)

                return graph 


            return _get_level_5_6
        

        if level ==7:
            def _get_level_7(n):
                f'directed grid. Hence shortest path may not be the right solution'
                m = int(math.sqrt(n))+1 #+1 because otherwise for n =10, we get a 3x3 grids, where the total amount of permutations are not enough. 
                # dimensions = m,m+random.randint(0,2)
                dimensions = m,m
                
                graph = nx.grid_2d_graph(*dimensions)
                start, end = (0, 0), (dimensions[0]-1, dimensions[1]-1)


                #generate all shortest paths and simple paths
                random.seed(10000*random.random())
                shortest_path = random.choice(list(nx.all_shortest_paths(graph, start, end)))
                all_paths = list(nx.all_simple_paths(graph, start, end))

                #only keep simple paths that are longer than shortest path
                
                shortest_path_length = len(shortest_path)
                all_other_paths = [path for path in all_paths if len(path) > shortest_path_length]

                

                #randomly select 3-5 paths from all paths, and remove the edges for the rest
                random.seed(10000*random.random())
                random.shuffle(all_other_paths)
                kept_paths = random.choices(all_other_paths, k=random.randint(3,5))

                #convert lists into correct format
                kept_paths_edges = {(path[i], path[i+1]) for path in kept_paths for i in range(len(path) - 1)}
                shortest_path_edges = {(shortest_path[i], shortest_path[i+1]) for i in range(len(shortest_path) - 1)}

                #take the union
                all_edges = list(kept_paths_edges | shortest_path_edges) #rename this


                #remove edges
                for edge in list(graph.edges):
                    if edge not in all_edges: graph.remove_edge(*edge)

                #remove any disconnected nodes. 
                graph = self._remove_disconnected_nodes(graph, start = (0,0))

                #lets randomly choose 2 paths from all_paths, that will be valid solutions. This doesnt have to include the shortest path necessarily. 
                random.shuffle(all_edges)
                
                forward_paths = random.choices(kept_paths, k=2)

                #populate a new directional graph. The forward paths have forward directioned arrows. For all other paths, arrows are randomly directioned. 
                graph = nx.DiGraph()
                covered_nodes = []
                for path in forward_paths:
                    # for u,v in path:
                    for i in range(len(path)-1):
                        u,v = path[i], path[i+1]
                        graph.add_edge(u,v)
                        covered_nodes.append((u,v))
                
                for path in all_edges:
                    for i in range(len(path)-1):
                        u,v = path[i], path[i+1]
                        if not (graph.has_edge(u,v) or graph.has_edge(v,u)):
                            #randomly select direction
                            (u,v) = (u,v) if random.random()<0.5 else (v,u)
                            graph.add_edge(u,v)
                

                #renaming the nodes. 
                nodes = sorted([node for node in graph.nodes], key= lambda x: m*x[0] + x[1])
                mapping = {node: k for k,node in enumerate(nodes)}
                
                graph = nx.relabel_nodes(graph, mapping)

                #total number of nodes change after removing nodes. 
                self.n = len(graph.nodes)

                for edge in graph.edges(): graph[edge[0]][edge[1]]['weight'] = random.randint(1, 5)

                return graph
            return _get_level_7
        
        if level==8:
            def _get_level_8(n):
                f'directed grid with no solution.'

                m = int(math.sqrt(n))+1 #+1 because otherwise for n =10, we get a 3x3 grids, where the total amount of permutations are not enough. 
                # dimensions = m,m+random.randint(0,2)
                dimensions = m,m
                
                graph = None
                graph = nx.grid_2d_graph(*dimensions)
                start, end = (0, 0), (dimensions[0]-1, dimensions[1]-1)


                #generate all shortest paths and simple paths
                random.seed(10000*random.random())
                shortest_path = random.choice(list(nx.all_shortest_paths(graph, start, end)))
                all_paths = list(nx.all_simple_paths(graph, start, end))

                #only keep simple paths that are longer than shortest path
                
                shortest_path_length = len(shortest_path)
                all_other_paths = [path for path in all_paths if len(path) > shortest_path_length]

                

                #randomly select 3-5 paths from all paths, and remove the edges for the rest
                random.seed(10000*random.random())
                random.shuffle(all_other_paths)
                kept_paths = random.choices(all_other_paths, k=random.randint(3,5))

                #convert lists into correct format
                kept_paths_edges = {(path[i], path[i+1]) for path in kept_paths for i in range(len(path) - 1)}
                shortest_path_edges = {(shortest_path[i], shortest_path[i+1]) for i in range(len(shortest_path) - 1)}

                #take the union
                all_edges = list(kept_paths_edges | shortest_path_edges) #rename this


                #remove edges
                for edge in list(graph.edges):
                    if edge not in all_edges: graph.remove_edge(*edge)

                #remove any disconnected nodes. 
                graph = self._remove_disconnected_nodes(graph, start = (0,0))

                #lets randomly choose 2 paths from all_paths, that will be valid solutions. This doesnt have to include the shortest path necessarily. 
                random.shuffle(all_edges)
                
                # forward_paths = random.choices(kept_paths, k=2)

                #populate a new directional graph. The forward paths have forward directioned arrows. For all other paths, arrows are randomly directioned. 
                graph = nx.DiGraph()
                # covered_nodes = []

                
                for path in all_edges:
                    for i in range(len(path)-1):
                        u,v = path[i], path[i+1]
                        if not (graph.has_edge(u,v) or graph.has_edge(v,u)):
                            #randomly select direction
                            (u,v) = (u,v) if random.random()<0.50 else (v,u)
                            graph.add_edge(u,v)
                

                #renaming the nodes. 
                nodes = sorted([node for node in graph.nodes], key= lambda x: m*x[0] + x[1])
                mapping = {node: k for k,node in enumerate(nodes)}
                
                graph = nx.relabel_nodes(graph, mapping)

                #total number of nodes change after removing nodes. 
                self.n = len(graph.nodes)

                for edge in graph.edges(): graph[edge[0]][edge[1]]['weight'] = random.randint(1, 5)

                return graph 


            return _get_level_8

        if level ==9:
            
            def _create_level_9(n):
                #we want 50% of paths returned to be eulerian and the remaining to be not. So that the model can evaluate fairly. 
                is_eulerian = random.choice([True, False])
                

                def create_eulerian_path_graph(nodes=n, edges=random.randint(n, 2*n), is_eulerian = is_eulerian):
                    
                    ret = None

                    if is_eulerian:
                        while True:
                            G = nx.gnm_random_graph(nodes, edges, seed = int(100000*random.random()))
                            odd_degree_nodes = [node for node, degree in dict(G.degree()).items() if degree % 2 == 1]

                            if (len(odd_degree_nodes) == 2 or len(odd_degree_nodes) == 0) and nx.is_connected(G):

                                ret = G
                                break
                    else:
                        while True:
                            G = nx.gnm_random_graph(nodes, edges, seed = int(100000*random.random()))
                            odd_degree_nodes = [node for node, degree in dict(G.degree()).items() if degree % 2 == 1]

                            if not (len(odd_degree_nodes) == 2 or len(odd_degree_nodes) == 0) and nx.is_connected(G):

                                ret = G
                                break
                    
                    return ret, is_eulerian
                

                graph, is_eulerian = create_eulerian_path_graph()


                start = None
                
                #checking which node to start from, such that there will exist a valid eulerian path. In case that odd_degree_nodes==2, you can't start from any node 
                if is_eulerian:
                    start = None
                    for i in list(graph.nodes):
                        try:
                            solution = list(nx.eulerian_path(graph, source = i))
                            start = solution[0][0]
                            break
                        except:
                            pass

                    graph = self._get_ordered_graph(graph, source = start)
                

                #start is the node to start from, which will lead to a eulerian path. If the graph is not Eulerian, then start =  None is returned. Else a valid node name is returned
                return graph, is_eulerian, start 
            return _create_level_9

        raise Exception(f'invalid Level number')
    
    
    
    def get_shortest_path(self, graph, source, target):
        shortest_path = nx.shortest_path(graph, source=source, target=target, weight='weight', method='dijkstra')
        ret = f''
        for node in shortest_path[:-1]: ret+=f'{node} -> '
        return ret+str(shortest_path[-1])
        

    def convert_adjArray_to_str(self, adj_array, mapping):
        new_line = '\n'
        out = f''''''
        out+= f"   {' '.join(mapping.values())}"
        for row_label, row in zip(mapping.values(), adj_array):
            out+=f'''{new_line} {row_label} {' '.join(map(str, row))}'''

        return out
    
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


