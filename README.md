# AI-Powered Cybersecurity Threat Detection

> End-to-end ML pipeline for multi-class network intrusion detection — Random Forest + Isolation Forest on the NSL-KDD benchmark dataset, with a real-time SOC dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-data--processing-150458?logo=pandas&logoColor=white)
![PowerBI](https://img.shields.io/badge/PowerBI-SOC--dashboard-F2C811?logo=powerbi&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Results at a Glance

| Metric | This Pipeline | Snort IDS (baseline) |
|---|---|---|
| Accuracy | **96.4%** | ~82% |
| AUC-ROC | **0.98** | 0.81 |
| False Positive Rate | **2.9%** | 17.2% |
| Detection classes | 5 (Normal + 4 attack types) | Signature-based |
| Training records | 125,000+ (NSL-KDD) | — |

---

## Architecture

```
NSL-KDD Dataset (125K+ records)
         │
         ▼
  [Data Preprocessing]
  StandardScaler + SMOTE
  (class imbalance correction)
         │
         ▼
  [Feature Engineering]
  src_bytes, dst_bytes, duration
  + 38 additional network features
         │
    ┌────┴────┐
    ▼         ▼
[Random Forest]  [Isolation Forest]
 5-class         Unsupervised anomaly
 classification  detection
 96.4% acc       FPR: 2.9%
 AUC: 0.98
    │
    ▼
[PowerBI SOC Dashboard]
Real-time KPIs + IP-level drill-down
```

---

## Attack Classes Detected

| Class | Description |
|---|---|
| Normal | Legitimate traffic |
| DoS | Denial of Service (SYN flood, Smurf, etc.) |
| Probe | Network scanning / reconnaissance |
| R2L | Remote-to-Local unauthorised access |
| U2R | User-to-Root privilege escalation |

---

## Tech Stack

| Component | Tool |
|---|---|
| Data processing | Pandas, NumPy |
| Class balancing | SMOTE (imbalanced-learn) |
| Supervised model | Random Forest (Scikit-learn) |
| Unsupervised model | Isolation Forest (Scikit-learn) |
| Validation | 5-fold stratified cross-validation |
| Visualisation | PowerBI, Matplotlib, Seaborn |
| Dataset | NSL-KDD (open source benchmark) |

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/mandativamshidhar/Cybersecurity_assesment_application.git
cd Cybersecurity_assesment_application
pip install -r requirements.txt
```

### 2. Run the training pipeline

```bash
python train.py
# Outputs: model accuracy, AUC, confusion matrix, feature importance plot
```

### 3. Run evaluation

```bash
python evaluate.py
# Outputs: classification report, ROC curve, FPR comparison vs Snort baseline
```

---

## Key Files

```
cybersecurity-threat-detection/
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis on NSL-KDD
│   ├── 02_preprocessing.ipynb    # SMOTE, scaling, feature engineering
│   └── 03_model_evaluation.ipynb # ROC curves, confusion matrix, FPR comparison
├── src/
│   ├── preprocess.py             # Data cleaning + SMOTE balancing
│   ├── train.py                  # Random Forest training + cross-validation
│   ├── anomaly.py                # Isolation Forest unsupervised detection
│   └── evaluate.py               # Metrics, AUC, confusion matrix
├── data/
│   └── README.md                 # NSL-KDD download instructions (not committed)
├── requirements.txt
└── README.md
```

---

## Feature Importance (Top 5)

The following features contributed most to attack classification:

| Rank | Feature | Importance |
|---|---|---|
| 1 | `src_bytes` | Bytes sent from source |
| 2 | `dst_bytes` | Bytes sent to destination |
| 3 | `duration` | Connection duration |
| 4 | `logged_in` | Login status flag |
| 5 | `count` | Connections to same host |

---

## SOC Dashboard

A PowerBI dashboard was built on top of model predictions, featuring:

- Real-time threat KPIs (attack rate, FPR, detection count by class)
- IP-level drill-down for investigating flagged connections
- Time-series trend of attack volume by type
- FPR comparison panel: Isolation Forest (2.9%) vs Snort IDS (17.2%)

---

## Why NSL-KDD?

NSL-KDD is the standard open-source benchmark for network intrusion detection research, fixing key issues in the original KDD Cup 1999 dataset (duplicate records, class imbalance). It enables direct comparison with published baselines.

---

## What I Learned

- SMOTE on a 5-class dataset requires per-class minority sampling strategy — naive SMOTE worsens majority-class accuracy
- Isolation Forest reduces FPR dramatically (17.2% → 2.9%) by flagging statistical outliers without needing labelled attack data
- Feature importance analysis showed that `src_bytes` and `dst_bytes` alone account for ~40% of model signal — network volume is the strongest attack indicator

---

## Related Work (Internship)

This project was developed during a Data Analyst internship at SSK Limited, Hyderabad (Jan–May 2026), focused on building an ML-driven alternative to signature-based IDS systems.

---

## Author

**Vamshidhar Reddy Mandati**  
AI/ML Engineer · [LinkedIn](https://linkedin.com/in/vamshidhar-reddy-mandati) · [GitHub](https://github.com/mandativamshidhar)

---

## License

MIT
