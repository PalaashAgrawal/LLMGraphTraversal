from graph_gen import create_level, get_graph_and_solution
import pandas as pd
from pathlib import Path

from tqdm import tqdm





def get_prompt(n, level, is_jumbled:bool = False, k=0):
    f'''
    n = order of graph
    level = level of complexity of the graph as defined in the graph_gen script
    is_jumbled = whether the graph nodes are jumbled in order
    k: whether the prompt is 0 shot, 1 shot or 3 shot
    '''
    assert n in [10,20], f'n (or order of graph) can only be 10 or 20'
    assert 1<=level<=10, f'invalid level'
    assert k in [0,1,3], f'invalid input for k. K (for k-shot prompting) should be one of 0,1, or 3'


    is_weighted = 'weighted' if level in [4,6,7,8,10] else 'unweighted'
    is_directed = 'directed' if level in [7,8] else 'undirected'
    goal_of_traversal = f'shortest' if is_weighted=='unweighted' else f'least cost'

#___________________________________________________k_shot prompts_________________________________________
    
    is_kshot = k>0
    k_shot_prompt = f''
    if is_kshot:
        if n ==10: k_shot_file =  Path('k_shot_prompts/o_10.csv')
        elif n==20 and not is_jumbled: k_shot_file = Path('k_shot_prompts/o_20.csv')
        elif n==20 and is_jumbled: k_shot_file = Path('k_shot_prompts/o_20_jumbled.csv')
        assert k_shot_file.exists()

        df = pd.read_csv(k_shot_file).drop('Unnamed: 0', axis=1)
        
        examples = df[df['level']==level].iloc[:k]

        k_shot_prompt = f'''Consider some examples'''

        
        for i, (_,example) in enumerate(examples.iterrows()):
            adjacency_matrix, solution, nodes_to_traverse = example['adjacency_matrix'], example['solution'], eval(example['nodes_to_traverse'])
            k_shot_prompt = k_shot_prompt + f'''\n\nExample {i+1}: {get_question(level, goal_of_traversal, nodes_to_traverse)}
{adjacency_matrix}

Solution: {solution}
        '''
        k_shot_prompt+=f'\n Given these examples, answer the following quesiton.'

#______________________________________________________actual prompt________________________________________
    
    definition_of_isweighted = f'the cost of travelling between the two nodes' if is_weighted=='weighted' else f'whether there is a connection between the two nodes'
    definition_of_adjacency_matrix = f'The value corresponding to each row M and column N represents {definition_of_isweighted}, where 0 means no connection'

    adjacency_matrix, solution, nodes_to_traverse, num_nodes = get_graph_and_solution(n, level, is_jumbled)
    # nodes_to_traverse = eval(nodes_to_traverse)


    #DO NOT CHANGE INDENTATION
    prompt = f''' Given is the adjacency matrix for a {is_weighted} {is_directed} graph containing {num_nodes +1} nodes labelled A to {chr(65+num_nodes)}. {definition_of_adjacency_matrix}.   

{k_shot_prompt}

{get_question(level, goal_of_traversal, nodes_to_traverse)}

{adjacency_matrix}
    '''


    return prompt, solution




def get_question(level, goal_of_traversal, nodes_to_traverse):

    if 1<=level<=8:
        question = f'what is the {goal_of_traversal} path from node {nodes_to_traverse[0]} to node {nodes_to_traverse[1]}? Return the sequence of nodes in response.'
    elif level==9:
        question = f'is the following a valid eulerian graph, if traversal is started from {nodes_to_traverse}? Return True or False in response.'
    elif level==10: 
        question = f'What is the {goal_of_traversal} path to travel first from node {nodes_to_traverse[0]} to node {nodes_to_traverse[1]}, and then from node node {nodes_to_traverse[1]} to node {nodes_to_traverse[2]}? Return 2 sequences of nodes in response.'

    return question 



# for level in range(1,11):
#     print(level)
#     prompt = get_prompt(n= 10,level = level, is_jumbled = False, k = 1)
#     print(prompt)

# level = 9
# for _ in tqdm(range(10)):
#     for k in [0,1,3]:
#         prompt = get_prompt(n= 20,level = level, is_jumbled = False, k = k)
        
#         # print(prompt)
        # prompt = get_prompt(n= 20,level = level, is_jumbled = True, k = k)
        # print(prompt)