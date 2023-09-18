from api import get_response, evaluate_response
from graph_gen import create_level


model = 'openai/gpt-4'
# model = 'openai/gpt-3.5-turbo'

graph_creator = create_level(6)
adj, shortest_path = graph_creator.create_graph(level = 1)

prompt = f'''
Given the adjacency graph below, what is the quickest path from {shortest_path[0]} to {shortest_path[-1]}?

{adj}
'''

print(prompt)
print(shortest_path)

response = get_response(prompt, model = model)
print(response)


is_correct = evaluate_response(response, shortest_path)
print(is_correct)





#_______________________
graph_creator = create_level(6, is_jumbled=True)
adj, shortest_path = graph_creator.create_graph(level=1)

prompt = f'''
Given the adjacency graph below, what is the quickest path from {shortest_path[0]} to {shortest_path[-1]}?

{adj}
'''

print(prompt)
print(shortest_path)
response = get_response(prompt, model = model)
print(response)



is_correct = evaluate_response(response, shortest_path)
print(is_correct)