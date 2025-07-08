# Prompt template for generating natural language queries based on table information
FORWARD_QUERY_PROMPT = """You are an amateur who doesn't understand data. Please construct possible query scenarios based on the data table information, describe the query problem in one sentence, and indicate the columns involved in this problem.
# Requirement document
'''
{requirement}
'''
# Data table information document:
'''
Table name: {table_name}

Introduction: {meta_description}

Terminology explanation:
{background_string}

Table column description: Each row represents a column information, with four fields separated by ;, format: column_name;column_chinese_name;column_explanation;possible_values.
{schema_string}
'''

{sample_string}
# Test sample {sample_count}:
Question:
"""

# Prompt template for generating SQL statements from natural language queries
FORWARD_SQL_PROMPT = """You are a professional SQL data engineer. Please understand the natural description statements describing query scenarios based on the data table information, and write SQL statements that can achieve their query requirements.

# Requirement document:
'''
{requirement}
'''

# Data table information document:
'''
Table name: {table_name}

Introduction: {meta_description}

Terminology explanation: {background_string}

Table column description: Each row represents a column information, with four fields separated by ;, format: column_name;column_chinese_name;column_explanation;possible_values.
{schema_string}
'''
{sample_string}
# Test sample {sample_count}:
Description statement: {query}
Generated SQL:
"""