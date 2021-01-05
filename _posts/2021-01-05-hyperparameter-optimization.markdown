---
layout: post
title:  "Hyperparameter optimization"
excerpt: "the common hyperparameter tuning methods are as following: Grid search, Random search and Bayesian optimization "
date:   2021-01-05 19:00:00
mathjax: true
---

In machine learning, hyperparameters (like learning rate, batch size, dropout rate, the number of layers and so on) need to be chosen deliberately before starting the training process,
which aims to get a model with the best performance.

**Grid Search** is simply an exhaustive searching through a manually specified subset of the hyperparameter space. A grid search algorithm must be guided by some performance metric, typically measured by cross-validation on the training set or evaluation on a held-out validation set.

**Random Search** replaces the exhaustive enumeration of all combinations by selecting them randomly.

**Bayesian Optimization** is a global optimization method for noisy black-box functions. Applied to hyperparameter optimization, Bayesian optimization builds a probabilistic model of the function mapping from hyperparameter values to the objective evaluated on a validation set. 
By iteratively evaluating a promising hyperparameter configuration based on the current model/observations, and then updating it, Bayesian optimization, aims to gather observations revealing as much information as possible about this function and, in particular, the location of the optimum. 
It tries to balance exploration (hyperparameters for which the outcome is most uncertain) and exploitation (hyperparameters expected close to the optimum). In practice, Bayesian optimization has been shown to obtain better results in fewer evaluations compared to grid search and random search, 
due to the ability to reason about the quality of experiments before they are run.
