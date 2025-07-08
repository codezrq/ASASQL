import random
import re
import pandas as pd
from sql_metadata import Parser
from openai import OpenAI
import faiss
from langchain.embeddings import HuggingFaceEmbeddings

from forward_datacenter import *
from prompt_hub import *
from data_generation.utils import *
import json

import argparse
parser = parser = argparse.ArgumentParser(description="Generate SQL according to the natural query")
parser.add_argument("--chat_model", help="the path of the chat model", type=str)
parser.add_argument("--input_path", help="the path of the input set", type=str)
parser.add_argument("--sql_per_query", help="Generate sql_per_query SQL queries for each natural query", type=int, default=10)
parser.add_argument("--output_path", help="the path of the output set", type=str)
args = parser.parse_args()


llm_client = OpenAI(api_key="empty", base_url="http://localhost:8000/v1")

def llm_chat(prompt):
    """
    Send a chat request to the language model
    
    Args:
        prompt (str): The input prompt
        
    Returns:
        str: The model's response
    """
    return llm_client.chat.completions.create(
        model="Qwen1.5-32B-Chat-GPTQ-Int4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=1.0,
        max_tokens=256,
        stop=["# Test sample"],  # Stop at test samples
    ).choices[0].message.content

SAMPLE_NUM = 3  # Number of test samples
gen_list = []  # List to store generation results


# Load existing data
with open('nl2sql-result.json', 'r', encoding='utf-8') as file:
    data_list = json.load(file)
    
for each in samples.keys():
    print(len(query_samples[each]))

def gen_sql(query, table):
    """
    Generate SQL statement for a given query and table
    
    Args:
        query (str): Natural language query
        table (str): Target table name
        
    Returns:
        tuple: (prompt, response) from the language model
    """
    meta_description = table_description[table]  # Table description
    selected_samples = random.sample(samples[table], SAMPLE_NUM)  # Randomly select SAMPLE_NUM examples
    example = ""
    
    # Build examples string
    for index, each in enumerate(selected_samples):
        example = example + f"# Test sample {index + 1}\nDescription statement: {each['query']}\nGenerated SQL: {each['sql']}\n"
    
    # Build schema string with column information
    schema_string = []  # Column descriptions
    flag = False  # Flag to mark if any column doesn't exist
    columns = column_name[table]
    random.shuffle(columns)  # Shuffle columns for diversity
    
    for col in columns:
        if col not in column_name[table]:  
            flag = True
            break
        schema_string.append(f"{col};{column_name_zh[table][col]};{column_description[table][col]};{json.dumps(possible_value[table][col],ensure_ascii=False)}")
    
    # If any column doesn't exist, use all columns from the table
    if flag == True:
        schema_string = []
        columns = column_name[table]
        random.shuffle(columns)
        for col in columns:
            schema_string.append(f"{col};{column_name_zh[table][col]};{column_description[table][col]};{json.dumps(possible_value[table][col],ensure_ascii=False)}")
    
    schema_string = '\n'.join(schema_string)
    
    # Get concepts involved in this table
    concepts = {}  # Concepts involved in the table
    for each in table_concept_name[table]:
        concepts[each] = concept_description[each]
    concepts = json.dumps(concepts, ensure_ascii=False)
    
    # Build the prompt
    prompt = FORWARD_SQL_PROMPT
    prompt = prompt.replace('{table_name}', table)
    prompt = prompt.replace('{meta_description}', meta_description)
    prompt = prompt.replace('{background_string}', concepts)
    prompt = prompt.replace('{schema_string}', schema_string)
    prompt = prompt.replace('{sample_string}', example)
    prompt = prompt.replace('{sample_count}', f"{SAMPLE_NUM + 1}")
    prompt = prompt.replace('{query}', query)
    print(prompt)
    response = llm_chat(prompt)
    return prompt, response

SQL_PER_QUERY = 100  # Number of SQL statements to generate per query
gen_list = []

    
count = 0
for each in data_list:
    count += 1
    query = each['origin_query']
    table = each['table']

    gen_list.append({"query": query,
                    "table": table})
    harsh = []  # List to track unique responses
    temp = 0
    
    # Generate multiple SQL statements for each query
    for i in range(SQL_PER_QUERY):
        temp += 1
        print(f"<<{count}.{i} send request")
        request, response = gen_sql(query, table)
        print(f">>{count}.{i} receive response")
        
        # Stop if we have 10 unique responses
        if len(harsh) == 10:
            break
            
        print(response)
        if response not in harsh:
            harsh.append(response)
            gen_list[-1][f"sql_{temp}"] = response

    
    # Save results every 10 iterations
    if count % 10 == 0:
        with open('nl2sql_val_sql.json', "w", encoding="utf-8") as json_file:
            json.dump(gen_list, json_file, ensure_ascii=False, indent=4)
    
# Final save of results
with open('nl2sql_val_sql.json', "w", encoding="utf-8") as json_file:
    json.dump(gen_list, json_file, ensure_ascii=False, indent=4)
            
