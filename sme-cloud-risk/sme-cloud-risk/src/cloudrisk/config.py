"""
Shared configuration for the cloudrisk pipeline.

Purpose : single source of truth for paths, the random seed and modelling
          constants so every stage is reproducible from one place.
Inputs  : none (constants only).
Outputs : module-level constants imported by the other stages.
"""

from pathlib import Path

# ---------------------------------------------------------------- Paths ------
# All paths are relative to the repository root so the project runs anywhere.
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

RAW_DATA_PATH = DATA_DIR / "sme_survey_raw.csv"
SAMPLE_DATA_PATH = DATA_DIR / "sample_sme_survey.csv"   # small committed sample

# ---------------------------------------------------------- Reproducibility --
SEED = 42

# ---------------------------------------------------------------- Modelling --
TARGET = "incident_12m"          # 1 if the SME reported a cloud incident in the last 12 months
TEST_SIZE = 0.25                 # stratified hold-out share
N_FOLDS = 5                      # stratified k-fold CV used for model selection
PRIMARY_METRIC = "roc_auc"       # metric that drives model selection
N_SHAP_BACKGROUND = 200          # background rows for the SHAP explainer

# Security controls surveyed. Each is a binary "in place" indicator and is the
# unit at which the final control-gap ranking is reported.
CONTROLS = [
    "mfa_enforced",
    "sso_in_use",
    "device_encryption",
    "endpoint_protection",
    "vpn_required",
    "least_privilege_access",
    "quarterly_access_review",
    "backup_tested",
    "security_training_annual",
    "incident_response_plan",
    "patching_sla",
    "shadow_it_policy",
]
