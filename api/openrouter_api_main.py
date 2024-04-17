import openai

openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = "" #replace with your own api key from openrouter.ai. 


headers = {
    "HTTP-Referer": "https://localhost"
}



# message = f'''
# Given the adjacency graph below, what is the quickest path from A to C?

#   A  B  C
# A 0, 1, 0, 
# B 1, 0, 1, 
# C 0, 1, 0

# return your answer in the following format:
# M->N->O
# '''

# message = f' Given the adjacency graph below, what is the quickest path from A to F? \n A B C D E F\n    A 0 1 0 0 0 0\n    B 1 0 1 0 0 0\n    C 0 1 0 1 0 0\n    D 0 0 1 0 1 0\n    E 0 0 0 1 0 1\n    F 0 0 0 0 1 0 \n\n output format: M, N, O'
# # message = f'what is the easiest way to make pancakes early in the morning?'

response = openai.ChatCompletion.create(
    # model="meta-llama/llama-2-13b-chat",  
    # model = 'google/palm-2-chat-bison',
    model = 'nousresearch/nous-hermes-llama2-13b',
    # model = "meta-llama/codellama-34b-instruct",  
    # model = "anthropic/claude-2",
    # model = "openai/gpt-3.5-turbo",	
    messages=[
    {"role": "user", "content": message}
    ],
    headers=headers  # Pass the headers to the API call
)
reply = response.choices[0].message



def get_response(message, model = "openai/gpt-3.5-turbo"):
    
  response = openai.ChatCompletion.create(
    # model="meta-llama/llama-2-13b-chat",  
    # model = 'google/palm-2-chat-bison',
    # model = 'nousresearch/nous-hermes-llama2-13b',
    # model = "meta-llama/codellama-34b-instruct",  
    # model = "anthropic/claude-2",
    # model = "openai/gpt-3.5-turbo",	
    model = model,
    messages=[{"role": "user", "content": message}],
    headers=headers  # Pass the headers to the API call
    )
  return response.choices[0].message
