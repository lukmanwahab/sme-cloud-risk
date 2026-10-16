"""
cloudrisk: predicting cloud security incident exposure for SMEs with remote workers.

Modules
-------
config    : paths, random seed and shared constants
simulate  : synthetic SME survey generator with a documented causal structure
features  : feature engineering from raw survey responses to a model matrix
train     : model benchmarking under stratified k-fold cross-validation
explain   : SHAP values and logistic odds ratios for control-gap ranking
evaluate  : hold-out evaluation, calibration and dashboard-ready exports
cli       : command-line entry point tying the stages together
"""

__version__ = "1.0.0"
