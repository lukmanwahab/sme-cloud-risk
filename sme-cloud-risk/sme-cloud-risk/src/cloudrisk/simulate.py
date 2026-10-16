"""
Synthetic SME cloud-security survey generator.

Purpose : the original CISST survey data is confidential, so this module
          generates a stand-in dataset with a documented causal structure.
          Firm characteristics drive which controls an SME adopts, and both
          characteristics and controls drive the probability of a cloud
          security incident. This makes the dataset realistic enough to
          exercise every stage of the pipeline (confounding, interaction
          effects, class imbalance, missing responses) while remaining fully
          transparent about how it was produced.
Inputs  : n_firms (int), seed (int).
Outputs : pandas DataFrame with one row per SME and the columns listed in
          COLUMN_DESCRIPTIONS; optionally written to CSV.
Depends : numpy, pandas.

Causal structure (summarised)
-----------------------------
sector, size, remote_share, it_staff, cloud_maturity  -->  controls
sector, size, remote_share, byod, controls            -->  incident_12m
Larger firms with dedicated IT staff adopt more controls; the same firms face
more attack surface. Controls reduce risk, with MFA, least-privilege access and
endpoint protection carrying the largest effects, mirroring the ordering found
in the applied study. Missing survey responses are injected at random in the
control questions to force the pipeline to handle them explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CONTROLS, RAW_DATA_PATH, SEED, TARGET

# Human-readable documentation of every generated column. Also used by the
# README generator and by the tests to confirm the schema is stable.
COLUMN_DESCRIPTIONS = {
    "firm_id": "Anonymised SME identifier",
    "sector": "Industry sector (categorical)",
    "employees": "Headcount (integer)",
    "remote_share": "Share of staff working remotely at least 3 days a week (0 to 1)",
    "it_staff": "Number of dedicated IT/security staff (integer)",
    "cloud_maturity": "Years since first cloud adoption (integer)",
    "byod_allowed": "1 if personal devices may access company cloud services",
    "n_cloud_services": "Number of distinct SaaS/cloud platforms in use",
    "prior_incident": "1 if the SME reported an incident more than 12 months ago",
    **{c: f"1 if control '{c}' is in place (binary, may be missing)" for c in CONTROLS},
    TARGET: "1 if a cloud security incident occurred in the last 12 months (target)",
}

SECTORS = ["professional_services", "retail", "manufacturing", "healthcare",
           "finance", "education", "technology", "construction"]

# Sector-level baseline risk on the logit scale (healthcare and finance are
# targeted more; construction and manufacturing have less cloud exposure).
SECTOR_RISK = {
    "professional_services": 0.10, "retail": 0.05, "manufacturing": -0.25,
    "healthcare": 0.35, "finance": 0.30, "education": 0.15,
    "technology": -0.10, "construction": -0.30,
}

# Protective effect of each control on the incident logit. Larger magnitude
# means a bigger reduction in incident probability when the control is present.
CONTROL_EFFECTS = {
    "mfa_enforced": -1.10,
    "least_privilege_access": -0.80,
    "endpoint_protection": -0.70,
    "security_training_annual": -0.55,
    "patching_sla": -0.50,
    "device_encryption": -0.40,
    "quarterly_access_review": -0.35,
    "vpn_required": -0.30,
    "incident_response_plan": -0.25,
    "sso_in_use": -0.25,
    "backup_tested": -0.15,
    "shadow_it_policy": -0.20,
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    return 1.0 / (1.0 + np.exp(-x))


def simulate_sme_survey(n_firms: int = 3000, seed: int = SEED,
                        missing_rate: float = 0.04) -> pd.DataFrame:
    """
    Generate a synthetic SME cloud-security survey.

    Parameters
    ----------
    n_firms      : number of SMEs to simulate.
    seed         : random seed for full reproducibility.
    missing_rate : share of control responses set to missing (item non-response).

    Returns
    -------
    DataFrame with one row per SME.
    """
    rng = np.random.default_rng(seed)

    # ----------------------------------------------------- Firm covariates --
    sector = rng.choice(SECTORS, size=n_firms)
    # UK SME definition: fewer than 250 employees. Log-normal gives the usual
    # long right tail of mostly micro and small firms.
    employees = np.clip(rng.lognormal(mean=2.8, sigma=1.0, size=n_firms), 1, 249).astype(int)
    log_emp = np.log(employees)
    remote_share = np.clip(rng.beta(2.0, 2.5, size=n_firms), 0, 1)
    # Dedicated IT staff scale with size; many micro firms have none.
    it_staff = rng.poisson(lam=np.clip(employees / 40.0, 0.05, 6.0))
    cloud_maturity = rng.integers(0, 12, size=n_firms)
    byod_allowed = rng.binomial(1, 0.45 + 0.3 * remote_share - 0.05 * np.log1p(it_staff))
    n_cloud_services = rng.poisson(lam=2 + 0.8 * log_emp + 3 * remote_share) + 1
    prior_incident = rng.binomial(1, 0.15)

    # --------------------------------------------------- Control adoption --
    # Propensity to adopt any control rises with size, IT staffing and cloud
    # maturity. Each control has its own intercept so adoption rates differ.
    adoption_base = (-0.8 + 0.35 * log_emp + 0.45 * np.log1p(it_staff)
                     + 0.08 * cloud_maturity + 0.6 * prior_incident)
    control_intercepts = rng.normal(0.0, 0.6, size=len(CONTROLS))
    controls = {}
    for j, name in enumerate(CONTROLS):
        p_adopt = _sigmoid(adoption_base + control_intercepts[j] + rng.normal(0, 0.3, n_firms))
        controls[name] = rng.binomial(1, p_adopt).astype(float)

    # ---------------------------------------------------- Incident outcome --
    # Attack surface grows with size, remote share, BYOD and service sprawl;
    # controls reduce it. An MFA x BYOD interaction reflects that MFA matters
    # most when personal devices are in play.
    logit = (1.1
             + np.vectorize(SECTOR_RISK.get)(sector)
             + 0.30 * log_emp
             + 0.90 * remote_share
             + 0.45 * byod_allowed
             + 0.08 * n_cloud_services
             + 0.50 * prior_incident
             - 0.04 * cloud_maturity)
    for name, effect in CONTROL_EFFECTS.items():
        logit += effect * controls[name]
    logit += -0.50 * controls["mfa_enforced"] * byod_allowed   # interaction
    logit += rng.normal(0, 0.35, n_firms)                        # unobserved heterogeneity
    incident = rng.binomial(1, _sigmoid(logit))

    # ------------------------------------------------------ Assemble frame --
    df = pd.DataFrame({
        "firm_id": [f"SME{idx:05d}" for idx in range(n_firms)],
        "sector": sector,
        "employees": employees,
        "remote_share": remote_share.round(3),
        "it_staff": it_staff,
        "cloud_maturity": cloud_maturity,
        "byod_allowed": byod_allowed,
        "n_cloud_services": n_cloud_services,
        "prior_incident": prior_incident,
        **controls,
        TARGET: incident,
    })

    # Inject item non-response in the control questions only. The outcome and
    # firm characteristics are treated as complete, as in the field survey.
    mask = rng.random((n_firms, len(CONTROLS))) < missing_rate
    df.loc[:, CONTROLS] = df[CONTROLS].mask(mask)

    return df


def write_dataset(n_firms: int = 3000, seed: int = SEED, path=RAW_DATA_PATH) -> pd.DataFrame:
    """Generate the survey and write it to CSV, creating the directory if needed."""
    df = simulate_sme_survey(n_firms=n_firms, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    frame = write_dataset()
    print(f"Wrote {len(frame):,} rows to {RAW_DATA_PATH}; incident rate = {frame[TARGET].mean():.3f}")
