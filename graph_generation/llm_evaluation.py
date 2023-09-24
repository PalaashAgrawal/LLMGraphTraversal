from api import get_response, evaluate_response
# from graph_gen import create_level
from prompt_generator import get_prompt


n = 10 
level = 2
is_jumbled = False
k=1

model = 'openai/gpt-3.5-turbo'
# model = 'openai/gpt-4'
# model = 'anthropic/claude-2'
# model = 'nousresearch/nous-hermes-llama2-13b'
# model = 'google/palm-2-chat-bison'


prompt, solution = get_prompt(n = n, level = level, is_jumbled=is_jumbled, k = k)
print(prompt)



response = get_response(prompt, model = model)
print(response)


is_correct = evaluate_response(response, solution)
print(is_correct)