import pandas as pd
import openpyxl
import re
import json

table_schema = pd.read_excel(
    "./data/description_full.xlsx", sheet_name="schema")

table_meta = pd.read_excel(
    "./data/description_full.xlsx", sheet_name="meta")

table_concept = pd.read_excel(
    "./data/description_full.xlsx", sheet_name="concept")


with open("./data/samples.json", 'r', encoding='utf-8') as file:
    cot_generated = json.load(file)
    


# table_sample

# Nested dictionary
# possible_value = {table_name: {column_name:[possible_value]}} # Possible values for columns
# column_name_zh = {table_name: {column_name: column_name_zh}} # Chinese names for columns
# column_description = {table_name: {column_name: description}} # Column descriptions
# table_column_info = {table_name: {column_name: "col; column_name_zh; column_description; possible_value"}} # Column information

# Dictionary
# column_name = {table_name: []} # Column names for each table
# fuzzy_column = {table_name: []} # Columns that need fuzzy querying
# sample = {table:['Question\nInvolved columns']} # Test samples
# column_info = {table_name:[col; column_name_zh; column_description; possible_value, ]}
# query_samples = {table:[]} # Test samples for query generation
# samples = {table:[]} # Test samples organized by table table_name: [json_line]
# concept_description = {entity: description}

# List
# table_list[i]: Name of the i-th table

possible_value = {}
column_name_zh = {}
column_name = {}
column_description = {}



for i in range(table_schema.shape[0]):
    if table_schema.iloc[i].reason != 'OK': # 只取那些reason值为OK的
        continue
    key1 = table_schema.iloc[i].table_name
    key2 = table_schema.iloc[i].column_name
    value1 = table_schema.iloc[i].column_name_zh
    value2 = table_schema.iloc[i].possible_value
    value3 = table_schema.iloc[i].description

    if isinstance(value1, str):
        value1.strip()
    if isinstance(value2, str):
        value2.strip()
    if isinstance(value3, str):
        value3.strip()
        
    if key1 not in possible_value:
        possible_value[key1] = {}
    if key1 not in column_name_zh:
        column_name_zh[key1] = {}
    if key1 not in column_name:
        column_name[key1] = []
    if key1 not in column_description:
        column_description[key1] = {}
    
    column_name[key1].append(key2)
    column_name_zh[key1][key2] = value1
    # print(value2)
    possible_value[key1][key2] = json.loads(value2)
    column_description[key1][key2] = value3
    
# table_list[i]: 第i个表的名称
table_list = []

table_description = {}

table_concept_name = {}
for i in range(table_meta.shape[0]):
    table_list.append(table_meta.iloc[i].table)
    table_description[table_meta.iloc[i].table] = table_meta.iloc[i].description
    table_concept_name[table_meta.iloc[i].table] = table_meta.iloc[i].concept_description.split(',')

# 字典 concept_description = {entity: description}
concept_description = {}

for i in range(table_concept.shape[0]):
    concept_description[table_concept.iloc[i].entity] = table_concept.iloc[i].description



# 字典
# column_info = {table_name:[col; column_name_zh; column_description; possible_value, ]}
column_info = {}
table_column_info = {}
for table in table_list:
    if table not in column_info:
        column_info[table] = []
    if table not in table_column_info:
        table_column_info[table] = {}
    for col in column_name[table]:
        column_info[table].append(f"{col};{column_name_zh[table][col]};{column_description[table][col]};{json.dumps(possible_value[table][col], ensure_ascii=False)}")
        # table_column_info[table][col] = f"{col};{column_name_zh[table][col]};{column_description[table][col]};{json.dumps(possible_value[table][col], ensure_ascii=False)}" # 
        table_column_info[table][col] = f"{col};{column_description[table][col]};{json.dumps(possible_value[table][col], ensure_ascii=False)}" # 
        
        

# query_samples = {table:[]}
query_samples = {}
samples = {}
for each in cot_generated:
    query = each['query']
    table = each['table']
    column = each['column']

    if table not in query_samples:
        query_samples[table] = []
    if table not in samples:
        samples[table] = []
    
    query_samples[table].append(f"Question: {query}\nInvolved columns: {column}")
    samples[table].append(each)
    

