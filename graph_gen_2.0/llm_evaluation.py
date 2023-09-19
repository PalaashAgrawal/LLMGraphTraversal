from api import get_response, evaluate_response
from graph_gen import create_level


model = 'openai/gpt-4'
# model = 'openai/gpt-3.5-turbo'

graph_creator = create_level(n=10, is_jumbled = True)
adj, shortest_path = graph_creator.create_graph(level = 2)



#TO DO, clearly write if we want the SHORTEST path, or the path with the least weight. Also mention if it is a directed/weighted graph. 
prompt = f'''
Given the adjacency graph below, what is the quickest path from {shortest_path[0]} to {shortest_path[-1]}?

{adj}
'''

print('prompt', prompt)
print('shortest_path', shortest_path)

response = get_response(prompt, model = model)
print(response)


is_correct = evaluate_response(response, shortest_path)
print(is_correct)



