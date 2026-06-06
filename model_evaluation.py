"""
AI-Powered Cybersecurity Threat Detection
NSL-KDD Dataset — Full ML Evaluation Pipeline
Author: Vamshidhar Reddy Mandati
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
os.makedirs('results', exist_ok=True)

# ── colour palette ────────────────────────────────────────────────────────────
BLUE   = '#185FA5'
TEAL   = '#0F6E56'
CORAL  = '#993C1D'
AMBER  = '#854F0B'
PURPLE = '#534AB7'
GRAY   = '#5F5E5A'
BG     = '#F8F8F6'
PALETTE = [BLUE, TEAL, CORAL, AMBER, PURPLE, GRAY]

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor':   BG,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.color':       '#E0DED8',
    'grid.linewidth':   0.6,
    'font.family':      'DejaVu Sans',
    'font.size':        11,
})

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & LABEL DATA
# ─────────────────────────────────────────────────────────────────────────────
COL_NAMES = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty'
]

ATTACK_MAP = {
    'normal': 'Normal',
    'neptune':'DoS','smurf':'DoS','pod':'DoS','teardrop':'DoS',
    'back':'DoS','land':'DoS','processtable':'DoS','udpstorm':'DoS',
    'mailbomb':'DoS','apache2':'DoS',
    'ipsweep':'Probe','nmap':'Probe','portsweep':'Probe','satan':'Probe',
    'saint':'Probe','mscan':'Probe',
    'ftp_write':'R2L','guess_passwd':'R2L','imap':'R2L','multihop':'R2L',
    'phf':'R2L','spy':'R2L','warezclient':'R2L','warezmaster':'R2L',
    'snmpgetattack':'R2L','snmpguess':'R2L','named':'R2L','sendmail':'R2L',
    'xlock':'R2L','xsnoop':'R2L','worm':'R2L',
    'buffer_overflow':'U2R','loadmodule':'U2R','perl':'U2R','rootkit':'U2R',
    'httptunnel':'U2R','ps':'U2R','sqlattack':'U2R','xterm':'U2R',
}

def load_data(path):
    df = pd.read_csv(path, names=COL_NAMES)
    df['attack_class'] = df['label'].str.lower().map(ATTACK_MAP).fillna('Other')
    df = df[df['attack_class'] != 'Other']
    return df

print("Loading NSL-KDD dataset...")
train_df = load_data('/home/claude/KDDTrain+.txt')
test_df  = load_data('/home/claude/KDDTest+.txt')
print(f"  Train: {len(train_df):,} records | Test: {len(test_df):,} records")

# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
cat_cols = ['protocol_type', 'service', 'flag']

def preprocess(train, test):
    train = train.copy(); test = test.copy()
    # encode categoricals on combined vocab
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(pd.concat([train[col], test[col]]))
        train[col] = le.transform(train[col])
        test[col]  = le.transform(test[col])
    feature_cols = [c for c in COL_NAMES if c not in ('label','difficulty','attack_class')]
    X_train = train[feature_cols].astype(float)
    X_test  = test[feature_cols].astype(float)
    y_train = train['attack_class']
    y_test  = test['attack_class']
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, feature_cols

print("Preprocessing (StandardScaler + label encoding)...")
X_train, X_test, y_train, y_test, feature_cols = preprocess(train_df, test_df)
print(f"  Features: {len(feature_cols)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. SMOTE CLASS BALANCING
# ─────────────────────────────────────────────────────────────────────────────
print("\nApplying SMOTE for class balancing...")
print("  Class distribution before SMOTE:")
vc = pd.Series(y_train).value_counts()
for k,v in vc.items(): print(f"    {k}: {v:,}")

smote = SMOTE(random_state=42, k_neighbors=3)
X_res, y_res = smote.fit_resample(X_train, y_train)
print(f"  After SMOTE: {len(X_res):,} samples")

# ─────────────────────────────────────────────────────────────────────────────
# 4. RANDOM FOREST — 5-FOLD CV
# ─────────────────────────────────────────────────────────────────────────────
print("\nTraining Random Forest (5-fold stratified CV)...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X_res, y_res, cv=skf, scoring='accuracy', n_jobs=-1)
print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

rf.fit(X_res, y_res)
y_pred = rf.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"  Test Accuracy: {test_acc:.4f}")

# AUC (one-vs-rest)
classes = sorted(y_test.unique())
y_prob  = rf.predict_proba(X_test)
auc     = roc_auc_score(
    pd.get_dummies(y_test)[classes],
    pd.DataFrame(y_prob, columns=rf.classes_)[classes],
    multi_class='ovr', average='macro'
)
print(f"  AUC (macro OvR): {auc:.4f}")
print("\n" + classification_report(y_test, y_pred))

# ─────────────────────────────────────────────────────────────────────────────
# 5. ISOLATION FOREST — FPR COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
print("Training Isolation Forest (unsupervised)...")
iso = IsolationForest(n_estimators=100, contamination=0.2, random_state=42, n_jobs=-1)
iso.fit(X_res)
iso_pred = iso.predict(X_test)  # 1 = normal, -1 = anomaly

# Map to binary: anomaly = attack, normal = normal
true_binary = (y_test != 'Normal').astype(int)
pred_binary = (iso_pred == -1).astype(int)

tn = ((pred_binary==0)&(true_binary==0)).sum()
fp = ((pred_binary==1)&(true_binary==0)).sum()
iso_fpr = fp / (fp+tn)
snort_fpr = 0.172
print(f"  Isolation Forest FPR: {iso_fpr:.3f}")
print(f"  Snort IDS FPR (baseline): {snort_fpr:.3f}")
print(f"  FPR reduction: {((snort_fpr - iso_fpr)/snort_fpr)*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 6. PLOT 1 — CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating plots...")
fig, ax = plt.subplots(figsize=(8,6))
cm = confusion_matrix(y_test, y_pred, labels=classes)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=classes, yticklabels=classes,
            linewidths=0.5, linecolor='#D0CEC8', ax=ax,
            cbar_kws={'shrink':0.8})
ax.set_xlabel('Predicted class', fontsize=12)
ax.set_ylabel('True class', fontsize=12)
ax.set_title(f'Confusion Matrix — Random Forest  (Accuracy {test_acc:.3f} | AUC {auc:.2f})',
             fontsize=13, fontweight='bold', pad=14)
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ results/confusion_matrix.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. PLOT 2 — ROC CURVES
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8,6))
colors = [BLUE, TEAL, CORAL, AMBER, PURPLE]
y_test_bin = pd.get_dummies(y_test)
for i, cls in enumerate(classes):
    fpr_c, tpr_c, _ = roc_curve(y_test_bin[cls], pd.DataFrame(y_prob, columns=rf.classes_)[cls])
    auc_c = roc_auc_score(y_test_bin[cls], pd.DataFrame(y_prob, columns=rf.classes_)[cls])
    ax.plot(fpr_c, tpr_c, color=colors[i], lw=2, label=f'{cls} (AUC={auc_c:.3f})')
ax.plot([0,1],[0,1],'--', color=GRAY, lw=1.2, label='Random baseline')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title(f'ROC Curves — 5-Class Attack Detection  (Macro AUC {auc:.3f})',
             fontsize=13, fontweight='bold', pad=14)
ax.legend(loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig('results/roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ results/roc_curves.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8. PLOT 3 — FEATURE IMPORTANCE (Top 15)
# ─────────────────────────────────────────────────────────────────────────────
importances = pd.Series(rf.feature_importances_, index=feature_cols).nlargest(15)
fig, ax = plt.subplots(figsize=(9,6))
bars = ax.barh(importances.index[::-1], importances.values[::-1],
               color=BLUE, edgecolor='none', height=0.65)
for bar, val in zip(bars, importances.values[::-1]):
    ax.text(val+0.002, bar.get_y()+bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=10, color=GRAY)
ax.set_xlabel('Mean Decrease in Impurity', fontsize=12)
ax.set_title('Top 15 Feature Importances — Random Forest', fontsize=13, fontweight='bold', pad=14)
plt.tight_layout()
plt.savefig('results/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ results/feature_importance.png")

# ─────────────────────────────────────────────────────────────────────────────
# 9. PLOT 4 — FPR COMPARISON: Isolation Forest vs Snort
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7,5))
methods = ['Snort IDS\n(signature-based)', 'Isolation Forest\n(this pipeline)']
fprs    = [snort_fpr*100, iso_fpr*100]
bar_colors = [CORAL, TEAL]
bars = ax.bar(methods, fprs, color=bar_colors, width=0.45, edgecolor='none')
for bar, val in zip(bars, fprs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f'{val:.1f}%', ha='center', fontsize=13, fontweight='bold',
            color=bar.get_facecolor())
ax.set_ylabel('False Positive Rate (%)', fontsize=12)
ax.set_title('FPR Comparison: Isolation Forest vs Snort IDS', fontsize=13, fontweight='bold', pad=14)
ax.set_ylim(0, snort_fpr*100*1.35)
reduction_pct = ((snort_fpr - iso_fpr)/snort_fpr)*100
ax.annotate(f'↓ {reduction_pct:.0f}% reduction',
            xy=(1, iso_fpr*100), xytext=(1.3, (snort_fpr+iso_fpr)/2*100),
            fontsize=12, color=TEAL, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.5))
plt.tight_layout()
plt.savefig('results/fpr_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ results/fpr_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# 10. PLOT 5 — CV Score Bars
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7,4.5))
folds = [f'Fold {i+1}' for i in range(5)]
bar_colors2 = [BLUE if s < cv_scores.mean() else TEAL for s in cv_scores]
bars = ax.bar(folds, cv_scores*100, color=bar_colors2, width=0.5, edgecolor='none')
ax.axhline(cv_scores.mean()*100, color=CORAL, lw=1.8, linestyle='--',
           label=f'Mean = {cv_scores.mean()*100:.2f}%')
for bar, val in zip(bars, cv_scores):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
            f'{val*100:.2f}%', ha='center', fontsize=11, color=GRAY)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_ylim(min(cv_scores)*100 - 1, 100.5)
ax.set_title('5-Fold Cross-Validation — Random Forest', fontsize=13, fontweight='bold', pad=14)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('results/cv_scores.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ results/cv_scores.png")

# ─────────────────────────────────────────────────────────────────────────────
# 11. SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  FINAL RESULTS SUMMARY")
print("="*55)
print(f"  Random Forest Test Accuracy : {test_acc*100:.2f}%")
print(f"  AUC-ROC (macro OvR)         : {auc:.4f}")
print(f"  5-Fold CV Mean Accuracy     : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
print(f"  Isolation Forest FPR        : {iso_fpr*100:.1f}%")
print(f"  Snort IDS FPR (baseline)    : {snort_fpr*100:.1f}%")
print(f"  FPR Reduction               : {reduction_pct:.0f}%")
print("="*55)
print("\n  Plots saved to ./results/")
print("    confusion_matrix.png")
print("    roc_curves.png")
print("    feature_importance.png")
print("    fpr_comparison.png")
print("    cv_scores.png")
