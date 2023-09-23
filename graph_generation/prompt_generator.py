from graph_gen import get_graph_and_solution



def get_prompt(n, level, is_jumbled:bool, k=0):
    f'''
    n = order of graph
    level = level of complexity of the graph as defined in the graph_gen script
    is_jumbled = whether the graph nodes are jumbled in order
    k: whether the prompt is 0 shot, 1 shot or 3 shot
    '''
    assert 1<=level<=10, f'invalid level'
    assert k in [0,1,3], f'invalid input for k. K (for k-shot prompting) should be one of 0,1, or 3'



   
    is_weighted = 'weighted' if level in [4,6,7,8,10] else 'unweighted'
    is_directed = 'directed' if level in [7,8] else 'undirected'
    
    
    definition_of_isweighted = f'the cost of travelling between the two nodes' if is_weighted else f'whether there is a connection between the two nodes'
    definition_of_adjacency_matrix = f'The value corresponding to each row M and column N represents {definition_of_isweighted}, where 0 means no connection'

    adjacency_matrix, solution, nodes_to_traverse, num_nodes = get_graph_and_solution(n, level, is_jumbled)

    goal_of_traversal = f'shortest' if is_weighted else f'least cost'


    prompt = f'''
    Given below is the adjacency matrix for a {is_weighted} {is_directed} graph containing {num_nodes +1} nodes labelled A to {chr(65+num_nodes)}. {definition_of_adjacency_matrix}.   
    
    what is the {goal_of_traversal} path from {nodes_to_traverse[0]} to {nodes_to_traverse[1]}?

    {adjacency_matrix}
    '''


    return prompt



get_prompt(10,5, is_jumbled = True, k = 0)

    