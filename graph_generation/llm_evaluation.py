from api import get_response, evaluate_response
from utils.get_model import get_model
from prompt_generator import get_prompt

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
result_dir = Path('results_prelim'); assert result_dir.exists()





n = 10 
level = 2
is_jumbled = False
k=1



f'''
model identifiers
gpt3.5
gpt4
hermes_llama2
claude2
palm
'''


identifiers = [
                # 'gpt3.5',
            #    'gpt4',
            #    'hermes_llama2',
               'claude2',
               'palm',
               ]




# df = pd.DataFrame(columns = {})

for model_id in identifiers:
    model = get_model(model_id)
    # model_dir = /
    # model_dir.mkdir(parents = True, exist_ok = True)
    print(model_id)


    for level in tqdm(range(1,11)):
        k_shot_options = [0,1,3]
        for k in k_shot_options:
            # options = ['o_10', 'o_20', 'o_20_jumbled']
            options = ['o_10'] #DELETE THIS LATER
            pandas_data = {option: {'prompt': [], 
                       'solution': [], 
                       'llm_response':[], 
                       'evaluator_response': [],

                       } for option in options
                        
                        }
            
            for option in options:
                if option=='o_10':
                    n=10
                    is_jumbled = False
                elif option =='o_20':
                    n=20
                    is_jumbled=False
                elif option=='o_20_jumbled':
                    n=20
                    is_jumbled =False
                

                prompt, solution = get_prompt(n, level, is_jumbled, k)
                pandas_data[option]['prompt'].append(prompt)
                pandas_data[option]['solution'].append(solution)
                response = get_response(prompt, model = model)
                pandas_data[option]['llm_response'].append(response)
                is_correct = evaluate_response(response, solution)
                pandas_data[option]['evaluator_response'].append(is_correct)
            

            res_dir = result_dir/f'{model_id}/level_{level}'
            res_dir.mkdir(parents = True, exist_ok= True)
            file = res_dir/f'k_{k}.xlsx'

            with pd.ExcelWriter(file, engine='xlsxwriter') as writer:
                for option in pandas_data:
                    data =pandas_data[option]
                    df = pd.DataFrame(data = data)
                    df.to_excel(writer, sheet_name = option, index = False)

        
               
