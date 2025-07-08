
#  modified from  https://github.com/KMnO4-zx/huanhuan-chat/blob/master/train.py

from datasets import Dataset
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
from peft import LoraConfig, TaskType, get_peft_model

import argparse
parser = parser = argparse.ArgumentParser(description="Finetune the reward model on the train set.")
parser.add_argument("--reward_model", help="the path of the reward model", type=str)
parser.add_argument("--train_set", help="the path of the train set", type=str)
parser.add_argument("--output_path", help="the path of the lora configuration", type=str)
args = parser.parse_args()

# System and user prompts for reward model training
sys_prompt = 'You are a professional SQL data engineer. Please judge whether the SQL and query semantics are consistent based on the given information'
user_prompt = """Based on the following field information, where table is the table information contained in SQL, formatted as '(table_name;table_explanation)'; column is the column information contained in SQL, separated by ',', each column information format is '(column_name;column_chinese_name;column_explanation;possible_values)'; query is a possible natural language description of SQL, which may be semantically inconsistent. Please judge whether the SQL and query semantics are consistent, only output 0 or 1. If the SQL and query semantics are consistent, output 1, otherwise output 0:
'''
table:{table}
column:{column}
SQL:{sql}
query:{query}
'''"""

def process_func(example):
    """
    Process a single example for training
    
    Args:
        example (dict): Training example with query, sql, table, column, and label
        
    Returns:
        dict: Processed example with input_ids, attention_mask, and labels
    """
    MAX_LENGTH = 256    # Llama tokenizer splits Chinese characters into multiple tokens, so we need to increase max length to ensure data integrity
    input_ids, attentionh_mask, labels = [], [], []
    
    # Build the prompt
    prompt = user_prompt
    prompt = prompt.replace('{sql}', example['sql'])
    prompt = prompt.replace('{query}', example['origin_query'].strip())
    prompt = prompt.replace('{table}', example['table'])
    prompt = prompt.replace('{column}', example['column'])
    
    # Tokenize the instruction and response
    instruction = tokenizer(f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)  # add_special_tokens doesn't add special tokens at the beginning
    response = tokenizer(f"{example['label']}", add_special_tokens=False)
    
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
tokenizer = AutoTokenizer.from_pretrained(args.reward_model, use_fast=False, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(args.reward_model, device_map="auto",torch_dtype=torch.bfloat16)
model.enable_input_require_grads()  # Required when enabling gradient checkpointing

# Load and process training dataset
train_path = args.train_set
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
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    logging_steps=10,
    num_train_epochs=12,
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
