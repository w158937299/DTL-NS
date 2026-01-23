# Dual-Tree LLM-Enhanced Negative Sampling for Implicit Collaborative Filtering

This repository contains the source code for the paper **"Dual-Tree LLM-Enhanced Negative Sampling for Implicit Collaborative Filtering"**.

## Requirements

- Python 3.10.13
- torch 2.9.0
- scikit-learn 1.6.1
- scipy 1.10.0
- transformers 4.57.3
- vllm 0.12.0
- numpy 1.26.4

## Usage

The workflow consists of four main steps:

1. **Dual Tree Construction & Item Encoding**  
   Run `encoding.py` to obtain item path encodings.

2. **False Negative Identification**  
   Execute `LLM_reasoning.py` to identify false negative.

3. **Enhanced Positive Sample Set Construction**  
   Run `sample_obtain.py` to build the enhanced Positive Sample Set.

4. **Model Training**  
   Finally, execute `main.py` to train the recommender.
