# AI-Based Cybersecurity Threat Detection System

> Interactive working model for the Industrial Training Report submitted to the Department of Computer Science & Engineering, Central University of Jammu.

**Student:** Mandati Vamshidhar Reddy (22BECCS23)  
**Company:** SSK Limited, Hyderabad  
**Mentor:** Ms. Tallapelli Ramya  
**Programme:** B.Tech CSE (Cyber Security)  
**Date:** May 2026

---

## Live Demo

Open `index.html` directly in any browser — no build step, no server required.

Or host it for free on GitHub Pages (see deployment section below).

---

## What this model demonstrates

The app is a front-end simulation of the complete ML pipeline described in the report.

| Tab | What it shows |
|-----|---------------|
| **Dashboard** | Live traffic trend (4 scenarios), attack distribution donut chart, real-time alert feed, Random Forest feature importance |
| **Classify event** | Enter 8 network features → get a Random Forest + Isolation Forest prediction with per-class probabilities |
| **Model comparison** | Performance table (Accuracy, Precision, Recall, F1, FPR, AUC) + ROC curve for all 4 models |
| **Confusion matrix** | Colour-coded 5×5 matrix from the NSL-KDD test set |

### Key metrics (from the report)

| Model | Accuracy | FPR | ROC-AUC |
|-------|----------|-----|---------|
| Random Forest | **96.4%** | **2.9%** | **0.98** |
| Logistic Regression | 91.3% | 8.4% | 0.92 |
| Isolation Forest | 88.7% | 5.6% | 0.91 |
| Snort IDS (baseline) | 82.1% | 17.2% | 0.81 |

---

## Technology stack

| Layer | Technology |
|-------|-----------|
| Front-end | HTML5 + CSS3 + Vanilla JS |
| Charts | Chart.js 4.4.1 (CDN) |
| Icons | Tabler Icons 2.44 (CDN) |
| ML models (report) | Python 3.9 · Scikit-learn 1.1 · Pandas · NumPy |
| Dataset | NSL-KDD (125,973 train / 22,544 test records) · CICIDS2017 |
| Dashboard (report) | PowerBI Desktop |

---

## Repository structure

```
ai-threat-detection/
├── index.html      ← entire app (single file, no dependencies to install)
├── README.md
└── .gitignore
```

---

## Run locally

```bash
# Clone
git clone https://github.com/<your-username>/ai-threat-detection.git
cd ai-threat-detection

# Just open the file — no npm install needed
open index.html          # macOS
start index.html         # Windows
xdg-open index.html      # Linux
```

---

## Deploy on GitHub Pages (free hosting)

```bash
# 1. Push to GitHub
git init
git add .
git commit -m "initial commit: AI cybersecurity threat detection model"
git branch -M main
git remote add origin https://github.com/<your-username>/ai-threat-detection.git
git push -u origin main

# 2. Enable GitHub Pages
# Go to: Settings → Pages → Source → Deploy from branch → main / (root) → Save
# Your site will be live at: https://<your-username>.github.io/ai-threat-detection/
```

---

## References (from report)

1. Tavallaee et al., "A Detailed Analysis of the KDD CUP 99 Data Set," IEEE CISDA, 2009.
2. Sharafaldin et al., "Toward Generating a New Intrusion Detection Dataset," ICISSP, 2018.
3. Breiman, "Random Forests," Machine Learning, vol. 45, 2001.
4. Liu et al., "Isolation Forest," IEEE ICDM, 2008.
5. Chawla et al., "SMOTE," Journal of AI Research, vol. 16, 2002.
6. Scikit-learn Developers, JMLR, vol. 12, 2011.

---

*This project was developed as part of the Data Analyst Internship at SSK Limited, Hyderabad.*
