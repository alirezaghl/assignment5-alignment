# solutions

- gsm8k baseline: run qwen 2.5 1.5B on gsm8k
    - code [gsm8k_baseline.py](https://github.com/alirezaghl/assignment5-alignment/blob/main/cs336_alignment/gsm8k_baseline.py)
    - code [baseline_results.py]([./cs336_alignment/outputs/math_baseline.md](https://github.com/alirezaghl/assignment5-alignment/blob/main/cs336_alignment/baseline_results.py))
- sft helper
    - code [sft_helper.py]([./cs336_alignment/utils.py](https://github.com/alirezaghl/assignment5-alignment/blob/main/cs336_alignment/sft_helper.py))



# CS336 Spring 2025 Assignment 5: Alignment

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment5_alignment.pdf](./cs336_spring2025_assignment5_alignment.pdf)

We include a supplemental (and completely optional) assignment on safety alignment, instruction tuning, and RLHF at [cs336_spring2025_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2025_assignment5_supplement_safety_rlhf.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

As in previous assignments, we use `uv` to manage dependencies.

1. Install all packages except `flash-attn`, then all packages (`flash-attn` is weird)
```
uv sync --no-install-package flash-attn
uv sync
```

2. Run unit tests:

``` sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

