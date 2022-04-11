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
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


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

def main():
    X_train, X_test, y_train, y_test = load_process_data("/hpf/largeprojects/adam/projects/timmy/lmp1210/files/data/proper_genex_data/Genex_ID_and_Peartree.tsv", "/hpf/largeprojects/adam/projects/timmy/lmp1210/files/data/proper_genex_data/redone_cpm.tsv")

    lr_search_space = {
        'feature_selection__n_components': [10, 20, 30, 35, 40, 100, 200],
        'clf__solver': ['lbfgs', 'liblinear'],
        'clf__penalty': ['l1', 'l2', 'elasticnet'],
        'clf__C': [1e-2, 1e-1, 1, 10, 100, 200, 300]
        }
    # lr_cv = gridsearch(lr, lr_search_space)
    # lr_cv.fit(X_train, y_train)

    xgb_search_space = {
        'feature_selection__n_components': [10, 20, 30, 35, 40, 100, 200],
        'clf__min_child_weight': [1, 5, 10],
        'clf__gamma': [0.1, 0.25, 0.5, 1, 1.5],
        'clf__subsample': [0.6, 0.8, 1.0, 1.2],
        'clf__colsample_bytree': [0.2, 0.4, 0.6],
        'clf__max_depth': [3, 4, 5]
        }
    # xgb_cv = gridsearch(xgb, xgb_search_space)
    # xgb_cv.fit(X_train, y_train)

    svc_search_space = {
        'feature_selection__n_components': [10, 20, 30, 35, 40, 100, 200],
        'clf__penalty': ["l1", "l2"],
        'clf__C': [1e-2, 1e-1, 1, 10, 100, 200, 300] 
        }
    # nb_cv = gridsearch(nb, nb_search_space)
    # nb_cv.fit(X_train, y_train)

    rf_search_space = {
        'feature_selection__n_components': [10, 20, 30, 35, 40, 100, 200],
        'clf__n_estimators': [300, 500, 750],
        'clf__max_features': ['auto', 'sqrt', 'log2'],
        'clf__max_depth' : [4,6,8,10,12],
        'clf__criterion' :['gini', 'entropy']
        }
    # rf_cv = gridsearch(rf, rf_search_space)
    # rf_cv.fit(X_train, y_train)
    names = [
            "Logistic_Regression",
            "XGBoost",
            "SVC",
            "Random_Forest"
        ]
    models = [
        LogisticRegression(max_iter=5000, random_state=0),
        XGBClassifier(random_state=0, use_label_encoder=False),
        LinearSVC(random_state=0),
        RandomForestClassifier(random_state=0)
        ]
    grids = [
        lr_search_space, 
        xgb_search_space, 
        svc_search_space, 
        rf_search_space
        ]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for name, model, grid in zip(names, models, grids):
        pipeline = Pipeline(
            [
                # ("feature_selection", SelectFromModel(SVC(kernel='linear'))),
                ("scaler", StandardScaler()),
                ("feature_selection", TruncatedSVD(random_state=0)),
                ("clf", model)
            ]
        )
        search = GridSearchCV(pipeline, param_grid = grid, cv=cv, scoring='f1', verbose=2, n_jobs=-1)
        search.fit(X_train, y_train)
        score = search.score(X_test, y_test)
        #save model
        pickle.dump(search, open(f"/hpf/largeprojects/adam/projects/timmy/lmp1210/files/models/{name}_final_s.pkl", "wb"))
if __name__ == "__main__":
    main()

X_train, X_test, y_train, y_test = load_process_data("/hpf/largeprojects/adam/projects/timmy/lmp1210/files/data/proper_genex_data/Genex_ID_and_Peartree.tsv", "/hpf/largeprojects/adam/projects/timmy/lmp1210/files/data/proper_genex_data/redone_cpm.tsv")

lr_search_space = {
    'feature_selection__n_components': [10, 20, 30, 35, 40, 100, 200],
    'clf__solver': ['lbfgs', 'liblinear'],
    'clf__penalty': ['l1', 'l2', 'elasticnet'],
    'clf__C': [1e-2, 1e-1, 1, 10, 100, 200, 300]
    }
# lr_cv = gridsearch(lr, lr_search_space)
# lr_cv.fit(X_train, y_train)

xgb_search_space = {
    'feature_selection__n_components': [10, 20, 30, 35, 40, 100, 200],
    'clf__min_child_weight': [1, 5, 10],
    'clf__gamma': [0.1, 0.25, 0.5, 1, 1.5],
    'clf__subsample': [0.6, 0.8, 1.0, 1.2],
    'clf__colsample_bytree': [0.2, 0.4, 0.6],
    'clf__max_depth': [3, 4, 5]
    }
# xgb_cv = gridsearch(xgb, xgb_search_space)
# xgb_cv.fit(X_train, y_train)

svc_search_space = {
    'feature_selection__n_components': [10, 20, 30, 35, 40, 100, 200],
    'clf__penalty': ["l1", "l2"],
    'clf__C': [1e-2, 1e-1, 1, 10, 100, 200, 300] 
    }
# nb_cv = gridsearch(nb, nb_search_space)
# nb_cv.fit(X_train, y_train)

rf_search_space = {
    'feature_selection__n_components': [10, 20, 30, 35, 40, 100, 200],
    'clf__n_estimators': [300, 500, 750],
    'clf__max_features': ['auto', 'sqrt', 'log2'],
    'clf__max_depth' : [4,6,8,10,12],
    'clf__criterion' :['gini', 'entropy']
    }
# rf_cv = gridsearch(rf, rf_search_space)
# rf_cv.fit(X_train, y_train)
names = [
        "Logistic_Regression",
        "XGBoost",
        "SVC",
        "Random_Forest"
    ]
models = [
    LogisticRegression(max_iter=5000, random_state=0),
    XGBClassifier(random_state=0, use_label_encoder=False),
    LinearSVC(random_state=0),
    RandomForestClassifier(random_state=0)
    ]
grids = [
    lr_search_space, 
    xgb_search_space, 
    svc_search_space, 
    rf_search_space
    ]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for name, model, grid in zip(names, models, grids):
    pipeline = Pipeline(
        [
            # ("feature_selection", SelectFromModel(SVC(kernel='linear'))),
            ("scaler", StandardScaler()),
            ("feature_selection", TruncatedSVD(random_state=0)),
            ("clf", model)
        ]
    )
    search = GridSearchCV(pipeline, param_grid = grid, cv=cv, scoring='f1', verbose=2, n_jobs=-1)
    search.fit(X_train, y_train)
    score = search.score(X_test, y_test)
    #save model
    pickle.dump(search, open(f"/hpf/largeprojects/adam/projects/timmy/lmp1210/files/models/{name}_final_s.pkl", "wb"))


fs = TruncatedSVD(n_components=100)
fs.fit(X_train, y_train)


pairtree = pd.read_csv("/hpf/largeprojects/adam/projects/timmy/lmp1210/files/data/proper_genex_data/Genex_ID_and_Peartree.tsv", sep="\t")
gene_exp = pd.read_csv("/hpf/largeprojects/adam/projects/timmy/lmp1210/files/data/proper_genex_data/redone_cpm.tsv", sep="\t")
gene_exp = gene_exp.set_index('genes').transpose()
gene_exp['sample'] = gene_exp.index.values
data = pairtree.merge(gene_exp, left_on="Secondary.ID", right_on='sample')
data["SBS31/35?"] = data["SBS31/35?"].map({'Y': 1, 'N': 0})
data = data.fillna(0)
features = data.iloc[:, 11:946].columns

best_features = [features[i] for i in fs.components_[0].argsort()[::-1]]