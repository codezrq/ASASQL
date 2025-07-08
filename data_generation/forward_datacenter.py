import pandas as pd
import openpyxl
import re
import json

# Load Excel data files
table_schema = pd.read_excel(
    "./data/description_full.xlsx", sheet_name="schema")

table_meta = pd.read_excel(
    "./data/description_full.xlsx", sheet_name="meta")

table_concept = pd.read_excel(
    "./data/description_full.xlsx", sheet_name="concept")

# Load sample data
with open("./data/samples.json", 'r', encoding='utf-8') as file:
    cot_generated = json.load(file)

# Data structure definitions:
# Nested dictionaries
# possible_value = {table_name: {column_name:[possible_value]}} # Possible values for columns
# column_name_zh = {table_name: {column_name: column_name_zh}} # Chinese names for columns
# column_description = {table_name: {column_name: description}} # Column descriptions
# table_column_info = {table_name: {column_name: "col; column_name_zh; column_description; possible_value"}} # Column information

# Dictionaries
# column_name = {table_name: []} # Column names for each table
# fuzzy_column = {table_name: []} # Columns that need fuzzy querying
# sample = {table:['question\ninvolved_columns']} # Test samples
# column_info = {table_name:[col; column_name_zh; column_description; possible_value, ]}
# query_samples = {table:[]} # Test samples for query generation
# samples = {table:[]} # Test samples organized by table table_name: [json_line]
# concept_description = {entity: description}

# Lists
# table_list[i]: Name of the i-th table

# Initialize data structures
possible_value = {}
column_name_zh = {}
column_name = {}
column_description = {}

# Process schema data to build column information
for i in range(table_schema.shape[0]):
    if table_schema.iloc[i].reason != 'OK':  # Only process columns with reason='OK'
        continue
    key1 = table_schema.iloc[i].table_name
    key2 = table_schema.iloc[i].column_name
    value1 = table_schema.iloc[i].column_name_zh
    value2 = table_schema.iloc[i].possible_value
    value3 = table_schema.iloc[i].description

    # Clean string values
    if isinstance(value1, str):
        value1.strip()
    if isinstance(value2, str):
        value2.strip()
    if isinstance(value3, str):
        value3.strip()

    # Initialize nested dictionaries if needed
    if key1 not in possible_value:
        possible_value[key1] = {}
    if key1 not in column_name_zh:
        column_name_zh[key1] = {}
    if key1 not in column_name:
        column_name[key1] = []
    if key1 not in column_description:
        column_description[key1] = {}
    
    # Store column information
    column_name[key1].append(key2)
    column_name_zh[key1][key2] = value1
    possible_value[key1][key2] = json.loads(value2)
    column_description[key1][key2] = value3

# Process table metadata
table_list = []  # List of table names
table_description = {}  # Table descriptions
table_concept_name = {}  # Concepts involved in each table

for i in range(table_meta.shape[0]):
    table_list.append(table_meta.iloc[i].table)
    table_description[table_meta.iloc[i].table] = table_meta.iloc[i].description
    table_concept_name[table_meta.iloc[i].table] = table_meta.iloc[i].concept_description.split(',')

# Process concept descriptions
concept_description = {}  # Dictionary of concept descriptions

for i in range(table_concept.shape[0]):
    concept_description[table_concept.iloc[i].entity] = table_concept.iloc[i].description

# Build column information strings
column_info = {}  # Column information organized by table
table_column_info = {}  # Detailed column information

for table in table_list:
    if table not in column_info:
        column_info[table] = []
    if table not in table_column_info:
        table_column_info[table] = {}
        
    for col in column_name[table]:
        column_info[table].append(f"{col};{column_name_zh[table][col]};{column_description[table][col]};{json.dumps(possible_value[table][col], ensure_ascii=False)}")
        table_column_info[table][col] = f"{col};{column_description[table][col]};{json.dumps(possible_value[table][col], ensure_ascii=False)}"  # Column name; description; possible values

# Process sample data
query_samples = {}  # Test samples for query generation
samples = {}  # Test samples organized by table

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

