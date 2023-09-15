from api import get_response, evaluate_response
from graph_gen import create_level


graph_creator = create_level(6)
adj, shortest_path = graph_creator.create_graph(1)

prompt = f'''
Given the adjacency graph below, what is the quickest path from {shortest_path[0]} to {shortest_path[-1]}?

{adj}
'''


# response = get_response(prompt)
# print(response)

response = f'The quickest path from A to F would be A -> B -> C -> E-> D -> F, with a total distance of 5.'

is_correct = evaluate_response(response, shortest_path)
print(is_correct)