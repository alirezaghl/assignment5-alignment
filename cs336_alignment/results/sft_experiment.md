## Question 1

**Perform supervised fine-tuning on reasoning examples from the MATH dataset (`/data/a5-alignment/MATH/sft.jsonl`) 
using the Qwen 2.5 Math 1.5B base model. The experiment varied the number of unique training examples across {128, 256, 512, 1024} 
and included training on the complete dataset. Hyperparameters (learning rate and batch size) were tuned to achieve a minimum validation accuracy of 15% on the full dataset.

### Results

The experiments demonstrate that using a batch size of 1024 yields approximately 18% validation accuracy. However, due to compute constraints, I was unable to complete training on the entire dataset and verify the final accuracy.

![SFT Validation Accuracy](https://github.com/alirezaghl/assignment5-alignment/blob/main/cs336_alignment/results/plots/sft_val_accuracy.png)

### Key Findings
- Batch size of 1024 achieved ~18% validation accuracy
- Complete dataset training remains incomplete

## Question 2:  
 Filter the reasoning SFT examples to only include examples that produce the correct answer. Run
 SFT on the (full) filtered dataset and report the size of the filtered dataset and the validation
 accuracy you achieve

**Status:** Incomplete due to insufficient computational resources.
