from api import get_response, evaluate_response
from graph_gen import create_level

model = 'openai/gpt-4'


def get_graph_and_solution(n, level, is_jumbled = False):
    f'given number of nodes, which level of graph to create and if the adjacency matrix has to be jumbled, get a valid adjacency matrix and solution'

    assert 1<=level<=10, f'invalid level'
    out, sh = f'', f''

    if 1<=level<=7 or level ==9:
        gen = create_level(n=n, is_jumbled=is_jumbled)
        out, sh, nodes_to_traverse = gen.create_graph(level = level)
    
    if level==8:
        while not sh.startswith('No'): #this is only for level 8

            gen = create_level(n=n, is_jumbled=is_jumbled)
            out, sh, nodes_to_traverse = gen.create_graph(level = level)
    
    if level==10:
        gen = create_level(n=n, is_jumbled=is_jumbled)
        out, sh, nodes_to_traverse = gen.create_graph(level = level)
        
        # return out, sh, nodes
    return out, sh, nodes_to_traverse






# graph_creator = create_level(n=10, is_jumbled = True)
# adj, shortest_path = graph_creator.create_graph(level = 2)



#TO DO, clearly write if we want the SHORTEST path, or the path with the least weight. Also mention if it is a directed/weighted graph. 
# prompt = f'''
# Given the adjacency graph below, what is the quickest path from {shortest_path[0]} to {shortest_path[-1]}?

# {adj}
# '''

# print('prompt', prompt)
# print('shortest_path', shortest_path)

# response = get_response(prompt, model = model)
# print(response)


# is_correct = evaluate_response(response, shortest_path)
# print(is_correct)