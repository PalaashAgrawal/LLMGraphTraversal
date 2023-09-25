def get_model(id):
    f'get model name from a simple identifier string'
    
    model_dict = {'gpt3.5':         'openai/gpt-3.5-turbo',
                  'gpt4':           'openai/gpt-4', 
                  'hermes_llama2':  'nousresearch/nous-hermes-llama2-13b', 
                  'claude2':        'anthropic/claude-2',
                  'palm':           'google/palm-2-chat-bison',

                  }
    if id in model_dict: return model_dict[id]
    else:
        # print(list(model_dict.items()))
        assert id in model_dict.values(), f'invalid identifier {id}'
    
    return id

