## Question

Implement expert iteration on the MATH training dataset (`/data/a5-alignment/MATH/train.jsonl`) using the Qwen 2.5 Math 1.5B Base model. The experiment explored variations in:
- Number of rollouts (G) per question
- Number of epochs in the SFT step
- Batch size for each expert iteration step (Db) from {512, 1024, 2048}

### Results

the experiment was conducted with an expert iteration batch size of 512. The model achieved approximately 80% validation accuracy after completing the expert iteration process.

![Expert Iteration Validation Accuracy](https://github.com/alirezaghl/assignment5-alignment/blob/main/cs336_alignment/results/plots/expert_val_accuracy.png)

### Summary
- **Batch size tested:** 512
- **Final validation accuracy:** ~80%
- **Limitation:** Additional batch sizes (1024, 2048) were not evaluated due to computational constraints
