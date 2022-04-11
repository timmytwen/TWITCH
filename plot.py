import numpy as np
import pandas as pd
import os
import sys
import pickle
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn_evaluation import plot

def load_process_data(pairtree_path, gene_exp_path):
    pairtree = pd.read_csv(pairtree_path, sep="\t")
    gene_exp = pd.read_csv(gene_exp_path, sep="\t")
    gene_exp = gene_exp.set_index('genes').transpose()
    gene_exp['sample'] = gene_exp.index.values
    data = pairtree.merge(gene_exp, left_on="Secondary.ID", right_on='sample')
    data["SBS31/35?"] = data["SBS31/35?"].map({'Y': 1, 'N': 0})
    data = data.fillna(0)
    X = data.iloc[:, 11:946].values
    y = data["SBS31/35?"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 0)
    #oversampling
    sm = SMOTE(random_state=0)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, X_test, y_res, y_test

def load_process_data_nogene(pairtree_path, gene_exp_path):
    pairtree = pd.read_csv(pairtree_path, sep="\t")
    gene_exp = pd.read_csv(gene_exp_path, sep="\t")
    gene_exp = gene_exp.set_index('genes').transpose()
    gene_exp['sample'] = gene_exp.index.values
    data = pairtree.merge(gene_exp, left_on="Secondary.ID", right_on='sample')
    data["SBS31/35?"] = data["SBS31/35?"].map({'Y': 1, 'N': 0})
    data = data.fillna(0)
    X = data.iloc[:, 11:23].values
    y = data["SBS31/35?"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 0)
    #oversampling
    sm = SMOTE(random_state=0)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, X_test, y_res, y_test


X_train, X_test, y_train, y_test = load_process_data("/hpf/largeprojects/adam/projects/timmy/lmp1210/files/data/proper_genex_data/Genex_ID_and_Peartree.tsv", "/hpf/largeprojects/adam/projects/timmy/lmp1210/files/data/proper_genex_data/redone_cpm.tsv")

lr = pickle.load(open("/hpf/largeprojects/adam/projects/timmy/lmp1210/files/models/Logistic_Regression.pkl","rb"))
xgb = pickle.load(open("/hpf/largeprojects/adam/projects/timmy/lmp1210/files/models/XGBoost.pkl","rb"))
nb = pickle.load(open("/hpf/largeprojects/adam/projects/timmy/lmp1210/files/models/Gaussian_Naive_Bayes.pkl","rb"))
rf = pickle.load(open("/hpf/largeprojects/adam/projects/timmy/lmp1210/files/models/Random_Forest.pkl","rb"))

lr_pred = lr.predict(X_test)
f1_score(y_test, lr_pred)
roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
print(classification_report(y_test, lr_pred))
plot.grid_search(lr.cv_results_, change=('clf__C', 'clf__penalty'),
                subset={'clf__solver': 'liblinear', 'clf__penalty': ["l1", "l2"]})

plt.tight_layout()
plt.savefig("/hpf/largeprojects/adam/projects/timmy/lmp1210/files/plots/lr_grid.png")


xgb_pred = xgb.predict(X_test)
f1_score(y_test, xgb_pred)
roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
print(classification_report(y_test, xgb_pred))


plt.cla()
plt.clf()
plot.grid_search(xgb.cv_results_, change=('clf__gamma', 'clf__min_child_weight'), 
    subset={'clf__colsample_bytree': 0.6, 'clf__max_depth': 3, "clf__subsample": 1.0})

plt.tight_layout()
plt.savefig("/hpf/largeprojects/adam/projects/timmy/lmp1210/files/plots/xgb_grid.png")


nb_pred = nb.predict(X_test)
f1_score(y_test, nb_pred)
roc_auc_score(y_test, nb.predict_proba(X_test)[:, 1])
print(classification_report(y_test, nb_pred))


rf_pred = rf.predict(X_test)
f1_score(y_test, rf_pred)
roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
print(classification_report(y_test, rf_pred))


plt.cla()
plt.clf()
plot.grid_search(rf.cv_results_, change=('clf__max_depth', "clf__n_estimators"), subset={"clf__criterion": "entropy", "clf__max_features": "log2"})

plt.tight_layout()
plt.savefig("/hpf/largeprojects/adam/projects/timmy/lmp1210/files/plots/nf_grid.png")




lr = pickle.load(open("/hpf/largeprojects/adam/projects/timmy/lmp1210/files/models/Logistic_Regression_final_s.pkl","rb"))
xgb = pickle.load(open("/hpf/largeprojects/adam/projects/timmy/lmp1210/files/models/XGBoost_final_s.pkl","rb"))
svc = pickle.load(open("/hpf/largeprojects/adam/projects/timmy/lmp1210/files/models/SVC_final_s.pkl","rb"))
rf = pickle.load(open("/hpf/largeprojects/adam/projects/timmy/lmp1210/files/models/Random_Forest_final.pkl","rb"))

X_train, X_test, y_train, y_test = load_process_data("/hpf/largeprojects/adam/projects/timmy/lmp1210/files/data/proper_genex_data/Genex_ID_and_Peartree.tsv", "/hpf/largeprojects/adam/projects/timmy/lmp1210/files/data/proper_genex_data/redone_cpm.tsv")

lr_pred = lr.predict(X_test)
f1_score(y_test, lr_pred)
roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
print(classification_report(y_test, lr_pred))

xgb_pred = xgb.predict(X_test)
f1_score(y_test, xgb_pred)
roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
print(classification_report(y_test, xgb_pred))

svc_pred = svc.predict(X_test)
f1_score(y_test, svc_pred)
roc_auc_score(y_test, svc.predict(X_test))
print(classification_report(y_test, svc_pred))


rf_pred = rf.predict(X_test)
f1_score(y_test, rf_pred)
roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
print(classification_report(y_test, rf_pred))

lr = pickle.load(open("/hpf/largeprojects/adam/projects/timmy/lmp1210/files/models/Logistic_Regression_final_nofs.pkl","rb"))
xgb = pickle.load(open("/hpf/largeprojects/adam/projects/timmy/lmp1210/files/models/XGBoost_final_nofs.pkl","rb"))
svc = pickle.load(open("/hpf/largeprojects/adam/projects/timmy/lmp1210/files/models/SVC_final_nofs.pkl","rb"))
rf = pickle.load(open("/hpf/largeprojects/adam/projects/timmy/lmp1210/files/models/Random_Forest_final_nofs.pkl","rb"))

X_train, X_test, y_train, y_test = load_process_data("/hpf/largeprojects/adam/projects/timmy/lmp1210/files/data/proper_genex_data/Genex_ID_and_Peartree.tsv", "/hpf/largeprojects/adam/projects/timmy/lmp1210/files/data/proper_genex_data/redone_cpm.tsv")

lr_pred = lr.predict(X_test)
f1_score(y_test, lr_pred)
roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
print(classification_report(y_test, lr_pred))

xgb_pred = xgb.predict(X_test)
f1_score(y_test, xgb_pred)
roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
print(classification_report(y_test, xgb_pred))

svc_pred = svc.predict(X_test)
f1_score(y_test, svc_pred)
roc_auc_score(y_test, svc.predict(X_test))
print(classification_report(y_test, svc_pred))


rf_pred = rf.predict(X_test)
f1_score(y_test, rf_pred)
roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
print(classification_report(y_test, rf_pred))
