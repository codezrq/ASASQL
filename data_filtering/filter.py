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
parser = parser = argparse.ArgumentParser(description="Select samples from data pool.")
parser.add_argument("--reward_model", help="the path of the reward model", type=str)
parser.add_argument("--input_path", help="the path of the input set", type=str)
parser.add_argument("--output_path", help="the path of the output set", type=str)
args = parser.parse_args()

SQL_PER_QUERY = 5


with open(args.input_path, 'r', encoding='utf-8') as file:
    data_list = json.load(file)



# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

sys_prompt = 'You are a professional SQL data engineer. Please judge whether the SQL and query semantics are consistent based on the given information'
user_prompt = """Based on the following field information, where table is the table information contained in SQL, formatted as '(table_name;table_explanation)'; column is the column information contained in SQL, separated by ',', each column information format is '(column_name;column_chinese_name;column_explanation;possible_values)'; query is a possible natural language description of SQL, which may be semantically inconsistent. Please judge whether the SQL and query semantics are consistent, only output 0 or 1. If the SQL and query semantics are consistent, output 1, otherwise output 0:
'''
table:{table}
column:{column}
SQL:{sql}
query:{query}
'''"""


def vllm_chat(request):
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
    prompt = user_prompt
    prompt = prompt.replace("{sql}", sql)
    prompt = prompt.replace("{query}", query)
    prompt = prompt.replace("{table}", table)
    prompt = prompt.replace("{column}", column)
    print(prompt)
    print(f"{msg}. send request!")
    response = vllm_chat(prompt)
    print(f"{msg}. recive request!")
    
    # Extract probabilities for 0 and 1
    last_token_logprobs = response.choices[0].logprobs.top_logprobs[0]
    # print(last_token_logprobs)
    if '0' in last_token_logprobs:
        prob_0 = math.exp(last_token_logprobs['0'])
    else:
        prob_0 = 0.000001
    if '1' in last_token_logprobs:
        prob_1 = math.exp(last_token_logprobs['1'])
    else:
        prob_1 = 0.000001

    prob = prob_0 + prob_1
    prob_0 = prob_0 / (prob)
    prob_1 = prob_1 / (prob)

    return prob_1

result_list = []
count = 0
for each in data_list:
    query = each['query']
    
    count += 1
    max_score = 0
    max_sql = ""
    max_table = ""
    max_column = ""
    for i in range(SQL_PER_QUERY):
        print(f"{count}.{i}")
        sql_key = f"sql_{i}"
        if sql_key not in each:
            continue
        sql = each[sql_key]
        # Invalid
        if "SELECT" not in sql or "FROM" not in sql:
            continue
        try: # Syntax error
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
        if table not in table_list: # Table name does not exist
            print(f"Table name does not exist{sql}")
            continue
        table_info = f"({table};{table_description[table]})"
        # Extract column name
        columns = [col.sql() for col in parsed.find_all(sqlglot.exp.Column)]
        # columns = extract_column(sql)
        columns = set(columns)
        columns.discard("NaN")
        columns = list(columns)
        if set(columns) - set(column_name[table]):
            print(f"Column name does not exist{sql} {set(columns) - set(column_name[table])}")
            continue
        column_info = []
        for col in columns:
            column_info.append(f"({col};{column_name_zh[table][col]};{column_description[table][col]};{possible_value[table][col]})")
        column_info = ",".join(column_info)
        # query, sql, table_info, column_info
        prob_1 = get_score(query, sql, table_info, column_info, f"{count}.{i}")
        # Record the highest score sql
        print(f"prob_1{prob_1}")
        if prob_1 > max_score:
            max_score = prob_1
            max_sql = sql
            max_table = table_info
            max_column = column_info
        
    result_list.append({"origin_query": query,
                        "sql": sql,
                        "table": max_table,
                        "column": max_column,
                        "prob_1": max_score})
    # Save as json file
    if count % 100 == 0:
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(result_list, f, ensure_ascii=False, indent=4)
        
with open(args.output_path, 'w', encoding='utf-8') as f:
    json.dump(result_list, f, ensure_ascii=False, indent=4)
            
        