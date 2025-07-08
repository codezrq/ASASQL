import json

import argparse
parser = parser = argparse.ArgumentParser(description="Generate SQL using the Best-of-N approach")
parser.add_argument("--input_path", help="the path of the input set", type=str)
parser.add_argument("--output_path", help="the path of the output set", type=str)
args = parser.parse_args()

# Load input data
with open(args.input_path, 'r', encoding='utf-8') as file:
    data_list = json.load(file)
    
# Select the maximum value from the top k results
def topK(data_list, k):
    """
    Select the best result from top k candidates for each query
    
    Args:
        data_list (list): List of data items with multiple SQL candidates
        k (int): Number of top candidates to consider
        
    Returns:
        list: List of best results for each query
    """
    result_list = []
    
    for each in data_list:
        temp_list = []
        
        # Collect all available SQL candidates up to k
        for i in range(k):
            if f"sql_{i}" not in each:
                continue
            if f"prob_{i}" not in each:
                each[f"prob_{i}"] = each[f'reward_{i}']
            temp_list.append({"origin_query": each['query'], 
                              "table": each['table'],
                              "32B_sql": each[f"sql_{i}"],
                              "reward_prob": each[f"prob_{i}"]})
        
        # Select the candidate with highest reward probability
        max_element = max(temp_list, key=lambda x:x['reward_prob'])
        result_list.append(max_element)
    return result_list

# Generate best-of-N results for different values of N
best_of_1 = result_list = topK(data_list, k=1)
best_of_3 = result_list = topK(data_list, k=3)
best_of_5 = result_list = topK(data_list, k=5)
best_of_7 = result_list = topK(data_list, k=7)
best_of_10 = result_list = topK(data_list, k=10)

# Filter results to ensure consistency across different N values
result_list = []
for i in range(len(best_of_1)):
    query_list = []
    query_list.append(best_of_1[i]['origin_query'])
    query_list.append(best_of_3[i]['origin_query'])
    query_list.append(best_of_5[i]['origin_query'])
    query_list.append(best_of_7[i]['origin_query'])
    query_list.append(best_of_10[i]['origin_query'])
    
    # Only keep results where all N values select the same query
    if len(list(set(query_list))) != 1:
        continue
    
    # Store the best-of-N results
    result_list.append({"origin_query": best_of_1[i]['origin_query'],
                        "table": best_of_1[i]['table'],
                        "best_of_1": best_of_1[i]['32B_sql'],
                        "reward_1": best_of_1[i]['reward_prob'],
                        "best_of_3": best_of_3[i]['32B_sql'],
                        "reward_3": best_of_3[i]['reward_prob'],
                        "best_of_5": best_of_5[i]['32B_sql'],
                        "reward_5": best_of_5[i]['reward_prob'],
                        "best_of_7": best_of_7[i]['32B_sql'],
                        "reward_7": best_of_7[i]['reward_prob'],
                        "best_of_10": best_of_10[i]['32B_sql'],
                       "reward_10": best_of_10[i]['reward_prob']})
    
print(len(result_list))

# Save results to output file
with open(args.output_path, 'w', encoding='utf-8') as file:
    json.dump(result_list, file, ensure_ascii=False, indent=4)
    
    
