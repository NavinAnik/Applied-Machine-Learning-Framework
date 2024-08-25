from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics
import pandas as pd
import os
import joblib

from . import dispatcher

FLOD = 0
TRAINING_DATA = os.environ.get("F://Codes//Applied-Machine-Learning-Framework//input//train_folds.csv")
MODEl = os.environ.get("model")

FLOD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}

if __name__ == "__main__":
    df = pd.read_csv("F://Codes//Applied-Machine-Learning-Framework//input//train_folds.csv")
    train_df = df[df.kfold.isin(FLOD_MAPPING.get(FLOD))]
    valid_df = df[df.kfold == FLOD]
    
    ytrain = train_df.target.values
    yvalid = valid_df.target.values
    
    train_df = train_df.drop(["id", "target", "kfold"], axis=1)
    valid_df = valid_df.drop(["id", "target", "kfold"], axis=1)
    
    valid_df = valid_df[train_df.columns]
    
    label_encoders = []
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist())
        train_df.loc[:, c] = lbl.transform(train_df[c].values)
        valid_df.loc[:, c] = lbl.transform(valid_df[c].values)
        label_encoders.append((c, lbl))
        
    clf = dispatcher.MODELS[MODEl]
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:, 1]
    print(metrics.roc_auc_score(yvalid, preds))  
    print(metrics.roc_auc_score(yvalid, preds))
    
    joblib.dump(label_encoders, f"F://Codes//Applied-Machine-Learning-Framework//models//{MODEl}_label_encoder.pkl")
    joblib.dump(clf, f"F://Codes//Applied-Machine-Learning-Framework//models//{MODEl}.pkl")
