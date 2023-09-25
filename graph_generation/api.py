import openai
from prompt_generator import get_prompt

openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = "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" #replace with your own api key. 


headers = {
    "HTTP-Referer": "https://localhost"
}



def get_response(message, model = "openai/gpt-3.5-turbo", timeout = 60):

    f'''
    feed in a message to a model through the openrouter API. Default model: GPT3.5 from openai. To see list of models, visit https://openrouter.ai/docs
    TO DO: create a function to get list of all models available (relevant to our project)
    '''
    
    response = openai.ChatCompletion.create(
    # model="meta-llama/llama-2-13b-chat",  
    # model = 'google/palm-2-chat-bison',
    # model = 'nousresearch/nous-hermes-llama2-13b',
    # model = "meta-llama/codellama-34b-instruct",  
    # model = "anthropic/claude-2",
    # model = "openai/gpt-3.5-turbo",	
    model = model,
    messages=[{"role": "user", "content": message}],
    headers=headers,  # Pass the headers to the API call
    request_timeout=timeout,
    )
    return response.choices[0].message["content"]



def evaluate_response(model_response, correct_response, model =  "openai/gpt-3.5-turbo", timeout = 60):
    f'''
    we will evaluate the response from LLMs using another small LLM (by default, GPT 3.5). 
    Input:
    model_response: response provided by the LLM 
    correct_response: ground truth for the shortest path. 
    '''

    prompt = f'''
    Given these two responses, the first given by a Langauge model, and the other is the ground truth response. 
    Evaluate if the answer provided by the language model is the same as the ground truth response. Answer in only one word -- "Correct" or "Wrong"
    
    Language model response: {model_response}

    ground truth response: {correct_response}
    '''


    response = openai.ChatCompletion.create(
    model = model,
    messages=[{"role": "user", "content": prompt}],
    headers=headers,  # Pass the headers to the API call
    request_timeout=timeout,
    )

    return response.choices[0].message["content"]

def evaluate_partial_response(model_response, correct_response, primary_evaluator_response:str ='', model = "openai/gpt-4", timeout = 60):
    f'''
    we will evaluate the response for partial correctness from LLMs using another small LLM (by default, GPT 3.5). 
    In our case, partial correctness is defined as how many nodes was the model able to get from the beginning. 
    THIS WILL ONLY RUN IF THE RESPONSE OF primary EVALUATOR (ie correct/wrong evaluator is correct)
    Input:
    model_response: response provided by the LLM 
    correct_response: ground truth for the shortest path. 
    ''' 
#Given a node sequence, how many nodes were predicted correctly by the language model before a wrong node is encountered. 




    if ('correct').lower() in primary_evaluator_response.lower(): return len(correct_response.split('->'))/len(correct_response.split('->'))




    prompt = f'''
    Given  two responses, the first given by a Langauge model, and the other is the ground truth response. 
    Evaluate the answer provided by the language model for partial correctness. Partial correctness is defined as follows.
    Given a node sequence, what fraction of the total nodes in the ground truth were predicted correctly by the language model before a wrong node is encountered.
    
    Here are a few examples.

    Example 1: 
    Language Model response: A -> B -> C -> E -> F
    ground truth response: A -> B -> D -> E -> F
    Output: 2/5
    Reason: The language model only got the first 2 of the total 5 nodes in the ground truth correctly (A and B) before predicting a wrong node.

    Example 2:
    Language Model response: A -> E -> D -> B
    ground truth response: A -> E -> D -> B
    Output: 4/4
    Reason: The language model only got all 4 out of 4 nodes of the ground truth correctly

    Example 3:
    Language Model response: A, D, C, F, H,  M
    ground truth response: A -> D -> C -> F -> H -> G -> M -> B 
    Output: 4/8
    Reason: The language model only got 4 nodes out of the total 8 node in the ground truth correctly before predicting a wrong node. 

    

    Given these examples, evaluate the following response from a langauge model. Return only the fraction as response, and not the reason, or the word "output"

    Language model response: {model_response}

    ground truth response: {correct_response}

    '''

    response = openai.ChatCompletion.create(
    model = model,
    messages=[{"role": "user", "content": prompt}],
    headers=headers,  # Pass the headers to the API call
    request_timeout=timeout,
    )

    return response.choices[0].message["content"]



# #testing


# prompt, solution = get_prompt(10,1)
# llm_response = get_response(prompt)
# is_correct = evaluate_response(llm_response, solution)
# partial_credit = evaluate_partial_response(llm_response, solution)


# get_response(prompt, timeout=1)

# print(prompt)
# print(solution)
# print(llm_response)

# print(is_correct)
# print(partial_credit)

