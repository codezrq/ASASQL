from openai import OpenAI
import json
import math
import csv

import argparse
parser = parser = argparse.ArgumentParser(description="Evaluate the reward model on the input set.")
parser.add_argument("--reward_model", help="the path of the reward model", type=str)
parser.add_argument("--input_path", help="the path of the input set", type=str)
parser.add_argument("--output_path", help="the path of the output set", type=str)
args = parser.parse_args()

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# System and user prompts for reward model evaluation
sys_prompt = 'You are a professional SQL data engineer. Please judge whether the SQL and query semantics are consistent based on the given information'
user_prompt = """Based on the following field information, where table is the table information contained in SQL, formatted as '(table_name;table_explanation)'; column is the column information contained in SQL, separated by ',', each column information format is '(column_name;column_chinese_name;column_explanation;possible_values)'; query is a possible natural language description of SQL, which may be semantically inconsistent. Please judge whether the SQL and query semantics are consistent, only output 0 or 1. If the SQL and query semantics are consistent, output 1, otherwise output 0:
'''
table:{table}
column:{column}
SQL:{sql}
query:{query}
'''"""

def vllm_chat(request):
    """
    Send a chat request to the vLLM server
    
    Args:
        request (str): The input request
        
    Returns:
        object: The model's response with logprobs
    """
    response = client.chat.completions.create(
        model=args.reward_model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": request},
        ],
        temperature=0,
        top_p=0.8,
        max_tokens=8,
        extra_body={
            "repetition_penalty": 1.05,
        },
        logprobs = True,
        top_logprobs=5
    )
    return response

result_list = []        

def evaluate(data_path, output_path):
    """
    Evaluate the reward model on the dataset
    
    Args:
        data_path (str): Path to the input dataset
        output_path (str): Path to save the evaluation results
    """
    # Load data
    with open(data_path, 'r', encoding='utf-8') as file:
        data_list = json.load(file)
        
    count = 0
    for each in data_list:
        count += 1
        prompt = user_prompt
        prompt = prompt.replace('{sql}', each['sql'])
        prompt = prompt.replace('{query}', each['origin_query'].strip())
        prompt = prompt.replace('{table}', each['table'])
        prompt = prompt.replace('{column}', each['column'])
        print(prompt, each['label'])
        print(f"<<{count}. send request!")
        response = vllm_chat(prompt)
        print(f">>{count}. receive response!")

        # Extract probabilities for 0 and 1
        last_token_logprobs = response.choices[0].logprobs.top_logprobs[0]
        
        # Calculate probability for '0'
        if '0' in last_token_logprobs:
            prob_0 = math.exp(last_token_logprobs['0'])
        else:
            prob_0 = 0.000001
            
        # Calculate probability for '1'
        if '1' in last_token_logprobs:
            prob_1 = math.exp(last_token_logprobs['1'])
        else:
            prob_1 = 0.000001
            
        # Normalize probabilities
        prob = prob_0 + prob_1
        prob_0 = prob_0 / (prob)
        prob_1 = prob_1 / (prob)  
        pred = "1" if prob_1 > 0.5 else "0"
        
        print(f">>{count}. compute prob: ")
        print(f"prob_0:{prob_0} prob_1:{prob_1}")
        
        # Store evaluation result
        result_list.append({
                            "origin_query": each['origin_query'],
                            "sql": each['sql'],
                            "table": each['table'],
                            "column": each['column'],
                            "prob_0": prob_0,
                           "prob_1": prob_1,
                           "pred": pred,
                           "label": each['label']})
        
    # Save results to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_list, f, ensure_ascii=False, indent=4)

print(args.input_path, args.output_path)
evaluate(args.input_path, args.output_path)




    