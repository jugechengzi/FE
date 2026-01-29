This repository provides tools for large-scale model editing using the MEMIT algorithm and our proposed FE-MEMIT method.

# Quick Start Guide

To run a complete editing pipeline, follow these steps in order. This example demonstrates how to precompute targets and then execute the edit.

## 1. Precompute Target Representations(z)

Before editing, compute the target hidden states (z). This is done once and can be reused for multiple runs.

- **all**: Calculates targets for every layer in EDITED_LAYERS via backpropagation.
- **firstforward** (FE Method): Calculates the target for the first layer via backpropagation, then uses a forward pass for the remaining layers.

```bash
Z_METHOD=firstforward
DATASET=multi_counterfact_20877
EDITED_LAYERS=4,5,6,7,8
MODEL=llama3-8b

python precompute_z.py --edited_layers=${EDITED_LAYERS} z_method=${Z_METHOD} data=${DATASET} num_z_samples=2000 seed=0 llms=${MODEL}
```

## 2. Run Model Editing

Once the targets are stored, execute the main editing script.


>IMPORTANT! To switch between the standard MEMIT and our FE-MEMIT, manually rename FE-memit_main.py to memit_main.py within the algs/memit/ directory before running the command below.



```bash
# Configuration
ALGO=memit
MODEL=llama3-8b
DATASET=multi_counterfact_20877
MODEL_NAME="llama3_memit_experiment"

python main.py \
  algs=${ALGO} \
  llms=${MODEL} \
  data=${DATASET} \
  save_name="${MODEL_NAME}" \
  num_edits=2000 \
  bs=2000 
```

## 3. Evaluation

After the model is edited, evaluate its performance on the target facts and its side effects on neighboring knowledge.

>Note: In locality test part, you may wish to manually change the following configurations:
 - In configs/config.yaml, change the defaults.llms to be the model you want to test.
 - In experiments/locality_exp_kl.py and experiments/locality_exp_topk.py, change the "data", "algs" and "save_names" under main function to match the results you wish to test.

```bash
# Standard Performance Test
python main.py algs=${ALGO} llms=${MODEL} data=${DATASET} test_only=True load_name="${MODEL_NAME}" save_name="${MODEL_NAME}"

# Neighbourhood Logits Generation
python main.py algs=${ALGO} llms=${MODEL} data=${DATASET} test_only=True neighborhood_logits=True load_name="${MODEL_NAME}" save_name="${MODEL_NAME}-neighborhood-logits"

# Locality Test
python experiments/locality_exp_kl.py
python experiments/locality_exp_topk.py
```

# Configuration Notes

>Custom Settings: For more advanced settings, please check *configs/config.yaml*.

Paths: You may wish to modify the default paths in the config for storing:
 - Edited model saved weight and Evaluation results. (results_dir)
 - Precomputed z vectors. (zs_cache_dir)


### Parameter Reference
 - bs : Batch size, number of data processed simultaneously
 - num_edits : Total number of data to be edited
 - z_method : Strategy for precomputing z
- test_only : Set to True to load a previously edited model for evaluation

`

