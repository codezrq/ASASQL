# ASASQL: Active Data Synthesis and Annotation for Efficient Natural Language SQL Generation in Cloud Systems


This repo is code for paper "ASASQL: Active Data Synthesis and Annotation for Efficient Natural Language SQL Generation in Cloud Systems". We are preparing more sample script and will update them soon.

During the experiment, we used `bge-large-zh-v1.5` as the embedding model and `qwen2.5-7B` as the base model.

## Structure

```
ASASQL/
├── data_generation/          # Stage 1: Data generation pipeline
├── activate_learning/        # Stage 2: Active learning framework
└── data_filtering/          # Stage 3: Data filtering and model training
```

## Usage 

#### Step 1: Install Dependencies
```bash
pip install torch transformers pandas openai faiss-cpu langchain
pip install sqlglot sql-metadata openpyxl
```

#### Step 2: Prepare Domain Data
Prepare your domain data in the `data/` folder according to the following format:

1. **data/description_full.xlsx** contains three sheets: schema, meta, and concept.

**Schema sheet** includes database table information:

| table_name | column_name | column_name_zh | description | reason | possible_value |
|------------|-------------|----------------|-------------|--------|----------------|
| host_details | host_vent | Host Name | Describes the host name of a host | OK | ["a", "b", "d"] |
| host_details | zone_name | Availability Zone | Records the availability zone name to which the host belongs. Use fuzzy matching when using this column | OK | ["zone1","zone2","zone3"] |

**Meta sheet** contains table descriptions:

| table | description |
|-------|-------------|
| host_details | Each row in the table records the attributes of a host, including host name, address... |

**Concept sheet** contains domain-specific terminology:

| entity | description |
|--------|-------------|
| region | A site is a data center set up in various locations, also known as a region, usually named with a city name and a number |

1. **data/data_sample.json** contains manually labeled training samples:

| key | value |
|-----|-------|
| origin_query | Natural language query |
| sql | SQL statement |
| table_name | Name of the table being queried |
| table | Table information, format: (table_name;table_description) |
| column | Column information, format: (column_name;column_chinese_name;column_description;possible_values) |
| label | Whether the SQL is syntactically correct and semantically consistent with the query |

### Stage 1: Data Generation

#### Step 1: Natural Query Generation
```bash
cd data_generation
python query_gen.py --chat_model <model_path> --embed_model <embed_path> --output_path <output_path>
```

This step generates natural language queries based on table schema information. The process:
- Uses the base model to generate queries from table descriptions
- Applies embedding-based deduplication to improve quality
- Outputs `forward_query.json` with generated queries and involved columns

#### Step 2: SQL Generation
```bash
python text2sql.py --chat_model <model_path>  --input_path <data_path> --sql_per_query <int> --output_path <output_path>
```

This step generates SQL statements for each natural language query:
- Generates multiple SQL variants per query for diversity
- Uses random column selection strategies
- Outputs `nl2sql-result.json` with (query, sql, table) tuples

#### Step 3: Table Column Information Extraction
```bash
python extract_sql.py --input_path <data_path> --output_path <output_path>
```

Extracts and formats table/column information from generated SQL statements for subsequent processing.

### Stage 2: Active Learning

#### Step 1: Evaluate Reward Model
```bash
cd activate_learning
python evaluation.py --reward_model <model_path> --input_path <data_path> --output_path <output_path>
```

Evaluates the base reward model on the generated data pool, assigning scores to each (query, sql) pair.

#### Step 2: Cluster
```bash
python cluster.py --reward_model <model_path> --input_path <data_path> --output_path <output_path>
```

Clusters candidate samples and removes redundant ones.

#### Step 3: Fine-tune Reward Model
```bash
python finetune.py --reward_model <model_path> --train_set <train_path> --output_path <output_path>
```
After annotation, fine-tunes the reward model using LoRA on the selected training samples.

#### Step 4: Iterate
Repeat steps 1-3 until the reward model performance meets requirement.

### Stage 3: Data Filtering and Model Training

#### Step 1: Filter High-Quality Samples
```bash
cd data_filtering
python filter.py --reward_model <model_path> --input_path <data_path> --output_path <output_path>
```

Uses the trained reward model to filter samples that are:
- Syntactically correct (passes SQL parsing)
- Semantically consistent (score above threshold)

#### Step 2: Fine-tunes NL2SQL Model
```bash
python finetune-nl2sql.py --base_model <model_path> --input_path <data_path> --output_path <output_path>
```

Fine-tunes a general language model on the filtered high-quality data to create the final NL2SQL model.




