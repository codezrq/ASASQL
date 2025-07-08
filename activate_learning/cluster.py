import json
from forward_datacenter import *
import sqlglot
from sqlglot import parse_one, exp
from data_generation.utils import *
from sqlglot.errors import ParseError
from openai import OpenAI
import json
import math

import argparse
parser = parser = argparse.ArgumentParser(description="Cluster-based deduplication of the dataset")
parser.add_argument("--reward_model", help="the path of the reward model", type=str)
parser.add_argument("--input_path", help="the path of the input set", type=str)
parser.add_argument("--output_path", help="the path of the output set", type=str)
args = parser.parse_args()

# Load input data
with open(args.input_path, 'r', encoding='utf-8') as file:
    data_list = json.load(file)

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

def get_score(query, sql, table, column, msg):
    """
    Get the reward score for a query-SQL pair
    
    Args:
        query (str): Natural language query
        sql (str): SQL statement
        table (str): Table information
        column (str): Column information
        msg (str): Message for logging
        
    Returns:
        float: Probability score for semantic consistency
    """
    prompt = user_prompt
    prompt = prompt.replace("{sql}", sql)
    prompt = prompt.replace("{query}", query)
    prompt = prompt.replace("{table}", table)
    prompt = prompt.replace("{column}", column)
    print(prompt)
    print(f"{msg}. send request!")
    response = vllm_chat(prompt)
    print(f"{msg}. receive request!")
    
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
    
    # Return probability for '1' (semantic consistency)
    return prob_1

result_list = []
count = 0

# Process each data item
for each in data_list:
    query = each['query']
    
    count += 1
    max_score = 0
    max_sql = ""
    max_table = ""
    max_column = ""
    
    # Evaluate multiple SQL statements for each query
    for i in range(5):
        print(f"{count}.{i}")
        sql_key = f"sql_{i}"
        if sql_key not in each:
            continue
        sql = each[sql_key]
        
        # Skip invalid SQL statements
        if "SELECT" not in sql or "FROM" not in sql:
            continue
            
        try:  # Check for syntax errors
            parsed = parse_one(sql)
        except ParseError as e:
            print("Parsing failed")
            continue
        except Exception as e:
            print(f"Parsing failed")
            continue
            
        # Extract table name
        tables = Parser(sql).tables
        tables = list(set(tables))
        table = tables[0]
        
        if table not in table_list:  # Check if table exists
            print(f"Table does not exist: {sql}")
            continue
            
        table_info = f"({table};{table_description[table]})"
        
        # Extract column names
        columns = [col.sql() for col in parsed.find_all(sqlglot.exp.Column)]
        columns = set(columns)
        columns.discard("NaN")
        columns = list(columns)
        
        # Check if all columns exist in the table
        if set(columns) - set(column_name[table]):
            print(f"Column does not exist: {sql} {set(columns) - set(column_name[table])}")
            continue
            
        # Build column information
        column_info = []
        for col in columns:
            column_info.append(f"({col};{column_name_zh[table][col]};{column_description[table][col]};{possible_value[table][col]})")
        column_info = ",".join(column_info)
        
        # Get score for this query-SQL pair
        prob_1 = get_score(query, sql, table_info, column_info, f"{count}.{i}")
        
        # Record the SQL with highest score
        print(f"prob_1{prob_1}")
        if prob_1 > max_score:
            max_score = prob_1
            max_sql = sql
            max_table = table_info
            max_column = column_info
        
    # Store the best result for this query
    result_list.append({"origin_query": query,
                        "sql": sql,
                        "table": max_table,
                        "column": max_column,
                        "prob_1": max_score})
    
    # Save results every 100 iterations
    if count % 100 == 0:
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(result_list, f, ensure_ascii=False, indent=4)
        
# Final save of results
with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(result_list, f, ensure_ascii=False, indent=4)
            
        
