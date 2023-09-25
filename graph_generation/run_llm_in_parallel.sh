#!/bin/bash

args=("gpt3.5" "gpt4" "hermes_llama2" "claude2" "palm")

for arg in "${args[@]}"; do
    nohup python llm_evaluation.py "$arg" &
done

echo "Evaluation done"