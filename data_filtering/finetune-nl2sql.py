#  modified from  https://github.com/KMnO4-zx/huanhuan-chat/blob/master/train.py



from datasets import Dataset
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
from peft import LoraConfig, TaskType, get_peft_model
from forward_datacenter import *

import argparse
parser = parser = argparse.ArgumentParser(description="Finetune the NL2SQL model.")
parser.add_argument("--base_model", help="the path of the base model", type=str)
parser.add_argument("--input_path", help="the path of the train set", type=str)
parser.add_argument("--output_path", help="the path of the lora config", type=str)
args = parser.parse_args()

# System and user prompts for NL2SQL model training
sys_prompt = 'You are a professional SQL data engineer. Please generate syntactically correct SQL statements that are semantically consistent with the natural query statements in the query field based on the given database table information. Please directly write the SQL statement corresponding to the query'

user_prompt = """
```# Data table information document:
Table name: {table}
Introduction: {table_info}
Field information:
Each row has four fields, separated by ';', which are column name, column Chinese name, column explanation, and possible values.
{columns}```
query:{query}
"""

def process_func(example):
    """
    Process a single example for NL2SQL training
    
    Args:
        example (dict): Training example with query, sql, table, and column information
        
    Returns:
        dict: Processed example with input_ids, attention_mask, and labels
    """
    MAX_LENGTH = 256    # Llama tokenizer splits Chinese characters into multiple tokens, so we need to increase max length to ensure data integrity
    input_ids, attentionh_mask, labels = [], [], []
    
    # Parse table information
    tables = example['table'][1:-1].split(';') 
    
    # Get column information for the table
    col_info = "\n".join(column_info[tables[0]])
    
    # Build the prompt
    prompt = user_prompt
    prompt = prompt.replace('{table}', tables[0])
    prompt = prompt.replace('{table_info}', tables[1])
    prompt = prompt.replace('{columns}', col_info)
    prompt = prompt.replace('{query}', example['origin_query'])
    
    print(prompt,'\n', example['sql'],'\n')
    
    # Tokenize the instruction and response
    instruction = tokenizer(f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)  # add_special_tokens doesn't add special tokens at the beginning
    response = tokenizer(f"{example['sql']}", add_special_tokens=False)
    
    # Combine instruction and response tokens
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # We also need to pay attention to the eos token, so set to 1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto",torch_dtype=torch.bfloat16)
model.enable_input_require_grads()  # Required when enabling gradient checkpointing

# Load and process training dataset
train_path = args.input_path
train_df = pd.read_json(train_path)
train_ds = Dataset.from_pandas(train_df)
train_set = train_ds.map(process_func, remove_columns=train_ds.column_names)

# Configure LoRA for parameter-efficient fine-tuning
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # Training mode
    r=64,  # LoRA rank
    lora_alpha=128,  # LoRA alpha, see LoRA principle for specific function
    lora_dropout=0.1  # Dropout ratio
)

# Apply LoRA to the model
peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()  # Print total trainable parameters

# Configure training arguments
args = TrainingArguments(
    output_dir=args.output_path,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    logging_steps=10,
    num_train_epochs=5,
    save_steps=50,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True
)

# Initialize trainer and start training
trainer = Trainer(
    model=peft_model,
    args=args,
    train_dataset=train_set,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
trainer.train()
