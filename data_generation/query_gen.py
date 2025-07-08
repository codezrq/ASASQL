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
import numpy as np

import argparse
parser = parser = argparse.ArgumentParser(description="Generate the natural query")
parser.add_argument("--chat_model", help="the path of the reward model", type=str)
parser.add_argument("--embed_model", help="the path of the embedding model", type=str)
parser.add_argument("--output_path", help="the path of the output set", type=str)
args = parser.parse_args()

THRESHOLD = 0.1  # Similarity threshold for deduplication
llm_client = OpenAI(api_key="sk-1234", base_url="http://localhost:8000/v1")

def llm_chat(prompt):
    """
    Send a chat request to the language model
    
    Args:
        prompt (str): The input prompt
        
    Returns:
        str: The model's response
    """
    return llm_client.chat.completions.create(
        model=args.chat_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=512,
        stop=["# Test sample"],  # Stop at test samples
    ).choices[0].message.content

# Initialize BGE embedding model for similarity calculation
bge_model = HuggingFaceEmbeddings(
    model_name=args.embed_model,
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)
vector_db = faiss.IndexFlatL2(1024)  # FAISS index for vector similarity search

SAMPLE_NUM = 3  # Number of test samples
gen_list = []  # List to store generation results
gen_full_list = []  # List to store full generation results

# Load existing query data for deduplication
with open('forward_query.json', 'r', encoding='utf-8') as file:
    temp = json.load(file)

# Process existing queries for deduplication
for each in temp:
    # Calculate embedding for the query
    query_v = np.array(bge_model.embed_documents([each['query']]))
    distance, index = vector_db.search(query_v, 1)
    
    # Skip if query is too similar to existing ones
    if distance[0][0] <= THRESHOLD:
        print("Removing duplicate sample")
        continue
    
    # Add query to vector database
    vector_db.add(query_v)
    query_samples[each['table']].append(f"Question: {each['query']}\nInvolved columns: {each['column']}")

# Print the number of samples for each table
for key in query_samples.keys():
    print(len(query_samples[key]))

# Main generation loop
i = 0
counter = 0
while 1:
    counter += 1
    table = random.choice(table_list)  # Select a random table
    description = table_description[table]  # Table description
    
    # Get concepts involved in this table
    concepts = {}
    for each in table_concept_name[table]:
        concepts[each] = concept_description[each]
    concepts = json.dumps(concepts, ensure_ascii=False)  # Concepts involved in the table
    
    column_info_str = '\n'.join(column_info[table])  # Column information string
    selected_sample = random.sample(query_samples[table], SAMPLE_NUM)  # Randomly select SAMPLE_NUM test samples for this table
    
    # Build examples string
    examples = ''
    for index, each in enumerate(selected_sample):
        examples = examples + f"# Test sample {index + 1}\n{each}\n"
        
    # Build the prompt
    prompt = FORWARD_QUERY_PROMPT
    prompt = prompt.replace('{table_name}', table)
    prompt = prompt.replace('{meta_description}', description)
    prompt = prompt.replace('{background_string}', concepts)
    prompt = prompt.replace('{schema_string}', column_info_str)
    prompt = prompt.replace('{sample_string}', examples)
    prompt = prompt.replace('{sample_count}', f"{SAMPLE_NUM + 1}")
    
    print(f"<<{counter}. send request.")
    response = llm_chat(prompt)
    print(f">>{counter}. receive response")
    
    # Parse the response to extract query and columns
    parts = response.split("Involved columns:")
    if len(parts) < 2:
        continue
    print(response)
    query = parts[0].strip()
    selected_column = parts[1].strip()
    print(query, "   ", selected_column)
    
    # Check for similarity with existing queries
    query_v = np.array(bge_model.embed_documents([query]))
    distance, index = vector_db.search(query_v, 1)
    if distance[0][0] <= THRESHOLD:
        print("Removing duplicate sample")
        continue
    
    # Add new query to vector database and samples
    vector_db.add(query_v)
    query_samples[table].append(f"Question: {query}\nInvolved columns: {selected_column}")
    
    # Store generation results
    gen_list.append({"request": prompt,
                     "response": response})
    
    gen_full_list.append({"query": query,
                          "table": table,
                          "column": selected_column})
    
    # Save results every 2 iterations
    if counter % 2 == 0:
        with open(args.output_path, "w", encoding="utf-8") as json_file:
            json.dump(gen_list, json_file, ensure_ascii=False, indent=4)


    


    
    

    

