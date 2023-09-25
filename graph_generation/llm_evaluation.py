from api import get_response, evaluate_response, evaluate_partial_response
from utils.get_model import get_model
from prompt_generator import get_prompt

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
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

identifiers = [sys.argv[1]] if len(sys.argv)>1 else [
                                                    # 'gpt3.5', #UNCOMMENT
                                                    # 'gpt4',
                                                    'hermes_llama2',
                                                    'claude2',
                                                    'palm',
                                                ]


# levels = range(1,11)
# levels = range(1,9) #FIRST WE DO LEVEL 1 to 8, because there's no error in that
# levels = range(9,11)
levels = [1,2,3,4,5,6,7,8,10]


for model_id in identifiers:
    print(model_id)
    model = get_model(model_id)

    for level in tqdm(levels, position = 0):
        k_shot_options = [0,1,3]
        for k in tqdm(k_shot_options, position = 1, leave = False):
            # options = ['o_10', 'o_20', 'o_20_jumbled'] #UNCOMMENT
            options = ['o_10']
            pandas_data = {option: {'prompt': [], 
                       'solution': [], 
                       'llm_response':[], 
                       'evaluator_response': [],
                       'evaluator_partial_correctness': [],
                       } for option in options
                        
                        }
            
            for option in tqdm(options, position =2, leave=False):
                if option=='o_10':
                    n=10
                    is_jumbled = False
                elif option =='o_20':
                    n=20
                    is_jumbled=False
                elif option=='o_20_jumbled':
                    n=20
                    is_jumbled =True
                
                #sometimes it gives random errors like AttributeError: choices in openAI api
                
                
                #FOR LOOP HERE for number of random examples

                prompt, solution = get_prompt(n, level, is_jumbled, k)
                

                #sometimes the prompt can exceed 4095 tokens. So we skip these using a simple heuristic. 
                if model_id in ['gpt3.5', 'hermes_llama2'] and len(prompt)>4500: continue
                
                try:
                    response = get_response(prompt, model = model)
                    is_correct = evaluate_response(response, solution)
                    # partial_correctness = evaluate_partial_response(response, solution, primary_evaluator_response = is_correct) if level not in [9,10] else 'N/A'
                    partial_correctness = evaluate_partial_response(response, solution) if level not in [9,10] else 'N/A'
                except:
                    response, is_correct, partial_correctness = f'', f'', f''
                
                # response, is_correct, partial_correctness = f'', f'', f''

                # try:

                #     response = get_response(prompt, model = model)
                # except Exception as e:
                #     response = 'error'

                # if response == 'error':
                #     is_correct = 'invalid input'
                #     partial_correctness = 'invalid input'
                # else: 
                #     try:
                #         is_correct = evaluate_response(response, solution)
                #     except Exception as e:
                #         is_correct = 'error'
                    
                #     try:
                #         partial_correctness = evaluate_partial_response(response, solution) if level not in [9,10] else 'N/A'
                #     except:
                #         partial_correctness = 'error'
                

                pandas_data[option]['prompt'].append(prompt)
                pandas_data[option]['solution'].append(solution)
                pandas_data[option]['llm_response'].append(response)
                pandas_data[option]['evaluator_response'].append(is_correct)
                pandas_data[option]['evaluator_partial_correctness'].append(partial_correctness)


                
                for data in pandas_data[option]:
                    assert len(pandas_data[option][data]) == len(pandas_data[option]['prompt'])

            
            res_dir = result_dir/f'{model_id}/level_{level}'
            res_dir.mkdir(parents = True, exist_ok= True)
            file = res_dir/f'k_{k}.xlsx'



            with pd.ExcelWriter(file, engine='xlsxwriter') as writer:
                for option in pandas_data:
                    data =pandas_data[option]
                    df = pd.DataFrame(data = data)
                    df.to_excel(writer, sheet_name = option, index = False)
            

        
               
