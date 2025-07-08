import json
from forward_datacenter import *
import sqlglot
from sqlglot import parse_one, exp
# from utils import *
from sqlglot.errors import ParseError
# from openai import OpenAI
import json
import math
import argparse
parser = parser = argparse.ArgumentParser(description="Extract the SQL from LLM response")
parser.add_argument("--input_path", help="the path of the train set", type=str)
parser.add_argument("--output_path", help="the path of the output set", type=str)
args = parser.parse_args()


def extract_column(sql, table):
    """
    Extract column information from SQL statement
    
    Args:
        sql (str): SQL statement to parse
        table (str): Table name for validation
        
    Returns:
        str: Comma-separated column information string
    """
    columns = []
    if "SELECT" not in sql or "FROM" not in sql:
        return columns
    try:
        parsed = parse_one(sql)
    except ParseError as e:
        return columns
    except Exception as e:
        return columns
    
    # Extract all column references from the SQL
    columns = [col.sql() for col in parsed.find_all(sqlglot.exp.Column)]
    columns = set(columns)
    columns.discard("NaN")
    columns = list(columns)
    
    # Build column information string
    column_infos = []
    for col in columns:
        if col in column_name[table]:
            column_infos.append(f"({col};{column_name_zh[table][col]};{column_description[table][col]};{possible_value[table][col]})")
            
    column_info = ",".join(column_infos)
    return column_info 

# Load query-SQL data for processing
with open(args.input_path, 'r', encoding='utf-8') as file:
    data_list = json.load(file)
       
result_list = []
for each in data_list:
    table = each['table']
    query = each['query']
    text = each['sql'] 
    
    # Extract SQL from markdown code blocks
    match = re.search(r"```SQL\s*(.*?)\s*```", text, re.DOTALL)

    if match:
        sql = match.group(1)
        # print(sql)
    else:
        print("SQL content not found")
        
    # Extract column information from SQL
    columns = extract_column(sql, table)
    print(columns, '\n')
    table_info = f"({table};{table_description[table]})"
    
    # Store processed result
    result_list.append({"table_name": table,
                        "table": table_info,
                        "column": columns,
                        "query": query,
                        "sql": sql})

# Save processed data to training file
with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(result_list, f, ensure_ascii=False, indent=4)
