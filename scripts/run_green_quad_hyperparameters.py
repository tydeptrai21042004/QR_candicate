"""Validation-only hyperparameter sensitivity study (separate from ablations)."""
from run_green_quad_ablation import main

if __name__ == '__main__':
    main(default_study='hyperparameters')
