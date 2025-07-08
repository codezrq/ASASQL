# Utility functions for ASASQL project
# This file contains common utility functions used across the project

import re
import json
from typing import List, Dict, Any

def clean_string(text: str) -> str:
    """
    Clean and normalize a string
    
    Args:
        text (str): Input string to clean
        
    Returns:
        str: Cleaned string
    """
    if isinstance(text, str):
        return text.strip()
    return text

def extract_table_name(sql: str) -> str:
    """
    Extract table name from SQL statement
    
    Args:
        sql (str): SQL statement
        
    Returns:
        str: Extracted table name
    """
    # Simple regex to extract table name from FROM clause
    match = re.search(r'FROM\s+(\w+)', sql, re.IGNORECASE)
    if match:
        return match.group(1)
    return ""

def validate_sql_syntax(sql: str) -> bool:
    """
    Basic SQL syntax validation
    
    Args:
        sql (str): SQL statement to validate
        
    Returns:
        bool: True if SQL appears valid, False otherwise
    """
    # Basic checks for common SQL keywords
    required_keywords = ['SELECT', 'FROM']
    sql_upper = sql.upper()
    
    for keyword in required_keywords:
        if keyword not in sql_upper:
            return False
    
    return True

def format_column_info(column_name: str, column_zh: str, description: str, possible_values: List[str]) -> str:
    """
    Format column information into standard string format
    
    Args:
        column_name (str): Column name
        column_zh (str): Chinese column name
        description (str): Column description
        possible_values (List[str]): Possible values for the column
        
    Returns:
        str: Formatted column information string
    """
    return f"{column_name};{column_zh};{description};{json.dumps(possible_values, ensure_ascii=False)}"

def parse_table_info(table_info: str) -> Dict[str, str]:
    """
    Parse table information string
    
    Args:
        table_info (str): Table information in format "(table_name;description)"
        
    Returns:
        Dict[str, str]: Parsed table information
    """
    if not table_info.startswith('(') or not table_info.endswith(')'):
        return {}
    
    content = table_info[1:-1]  # Remove parentheses
    parts = content.split(';', 1)
    
    if len(parts) >= 2:
        return {
            'table_name': parts[0],
            'description': parts[1]
        }
    return {}

def parse_column_info(column_info: str) -> List[Dict[str, str]]:
    """
    Parse column information string
    
    Args:
        column_info (str): Column information in format "(col1;zh1;desc1;vals1),(col2;zh2;desc2;vals2)"
        
    Returns:
        List[Dict[str, str]]: Parsed column information
    """
    columns = []
    if not column_info:
        return columns
    
    # Split by ),( to get individual column entries
    parts = column_info.split('),(')
    
    for part in parts:
        # Remove outer parentheses
        if part.startswith('('):
            part = part[1:]
        if part.endswith(')'):
            part = part[:-1]
        
        # Split by ; to get column components
        col_parts = part.split(';')
        if len(col_parts) >= 4:
            columns.append({
                'column_name': col_parts[0],
                'column_zh': col_parts[1],
                'description': col_parts[2],
                'possible_values': col_parts[3]
            })
    
    return columns 