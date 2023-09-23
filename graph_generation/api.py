import openai

openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = "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" #replace with your own api key. 


headers = {
    "HTTP-Referer": "https://localhost"
}



def get_response(message, model = "openai/gpt-3.5-turbo"):

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
    headers=headers  # Pass the headers to the API call
    )
    return response.choices[0].message["content"]



def evaluate_response(model_response, correct_response, model =  "openai/gpt-3.5-turbo"):
    f'''
    we will evaluate the response from LLMs using another small LLM (by default, GPT 3.5). 
    Input:
    model_response: response provided by the LLM 
    correct_response: ground truth for the shortest path. 
    '''

    prompt = f'''
    Given these two responses, the first given by a Langauge model, and the other is the ground truth response. 
    Evaluate if the answer provided in the language model is the same as the ground truth response. Answer in only one word -- "Correct" or "Wrong"
    
    Language model response: {model_response}

    ground truth response: {correct_response}
    '''


    response = openai.ChatCompletion.create(
    model = model,
    messages=[{"role": "user", "content": prompt}],
    headers=headers  # Pass the headers to the API call
    )

    return response.choices[0].message["content"]