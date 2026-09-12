%pip install seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("customer_churn_dataset-testing-master.csv")

df.info()

df=df.drop(columns=["CustomerID"],errors="ignore")

from sklearn.compose import make_column_selector
string_cols=make_column_selector(dtype_include=["object","category"])(df)
encoded=pd.get_dummies(df,columns=string_cols,dtype=int)

encoded.info()

X=encoded.drop("Churn",axis=1)
y=encoded["Churn"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
models={"Decision Tree":DecisionTreeClassifier(max_depth=5),"Random Forest":RandomForestClassifier(),"Gradient Boosting":GradientBoostingClassifier()}

for name,model in models.items():
    model.fit(X_train_scaled,y_train)
    y_pred=model.predict(X_test_scaled)
    y_pred_proba=model.predict_proba(X_test_scaled)[:,1]
    from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix,roc_auc_score
    
    precision=precision_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    confusion=confusion_matrix(y_test,y_pred)
    roc=roc_auc_score(y_test,y_pred_proba)
    print("Precision:",precision)
    print("Recall:",recall)
    print("F1-score:",f1)
    print("Confusion Matrix:",confusion)
    print("ROC-AUC:",roc)
