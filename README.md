# SmoothBandit

Code for the SimplifiedSmoothBandit algorithm in "Smooth Contextual Bandits: Bridging the Parametric and Non-differentiable Regret Regimes"
https://arxiv.org/abs/1909.02553

To produce Figures 5:
- Run `python makedata.py b` for b=1.5,5.5 to generate a contextual bandit instances
- Run `python experiment.py i j s` for i=0,1, j=0,...,10, and s=0,...,4 to run different bandit algorithm (j) on problems with different smoothness (i) and using different seeds (s)
- Use `experiment results.ipynb` to generate plots
