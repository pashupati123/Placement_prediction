import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

default=pd.read_csv('default.csv')
default.rename(columns=lambda x: x.lower(),inplace=True)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,precision_recall_curve
from sklearn.preprocessing import RobustScaler

target_name='default'
X=default.drop('default',axis=1)
robust_scaler=RobustScaler()
X=robust_scaler.fit_transform(X)
y=default[target_name]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,random_state=123,stratify=y)

def CMatrix(CM,labels=['placed','Not-Placed ']):
    df=pd.DataFrame(data=CM,index=labels,columns=labels)
    df.index.name='TRUE'
    df.columns.name='PREDICTION'
    df.loc['Total']=df.sum()
    df['Total']=df.sum(axis=1)
    return df


metrics=pd.DataFrame(index=['accuracy','precision','recall'],
                     columns=['NULL','LogisticReg','ClassTree','NaiveBayes'])


y_pred_test=np.repeat(y_train.value_counts().idxmax(),y_test.size)
metrics.loc['accuracy','NULL']=accuracy_score(y_pred=y_pred_test,y_true=y_test)
metrics.loc['precision','NULL']=precision_score(y_pred=y_pred_test,y_true=y_test)
metrics.loc['recall','NULL']=recall_score(y_pred=y_pred_test,y_true=y_test)
CM=confusion_matrix(y_pred=y_pred_test,y_true=y_test)

print("MODEL0")
print("Confusion Matrix for NULL MODEL")
print("--------------------------------------------------")

print(CMatrix(CM))
print("--------------------------------------------------")
print("MODEL1")
print("Confusion Matrix for LOGISTICREGRESSION MODEL")
print("--------------------------------------------------")


#LogisticRegression Model

from sklearn.linear_model import LogisticRegression
logistic_regression=LogisticRegression(n_jobs=-1,random_state=15)
logistic_regression.fit(X_train,y_train)
y_pred_test=logistic_regression.predict(X_test)
metrics.loc['accuracy','LogisticReg']=accuracy_score(y_pred=y_pred_test,y_true=y_test)
metrics.loc['precision','LogisticReg']=precision_score(y_pred=y_pred_test,y_true=y_test)
metrics.loc['recall','LogisticReg']=recall_score(y_pred=y_pred_test,y_true=y_test)
CM=confusion_matrix(y_pred=y_pred_test,y_true=y_test)
print(CMatrix(CM))



#Classifier with threshod of 0.2
'''print("---------------------------------------------------------------")
print("Classifier with threshold of 0.2")


y_pred_proba=logistic_regression.predict_proba(X_test)[:,1]
y_pred_test=(y_pred_proba>=0.2).astype('int')

CM=confusion_matrix(y_pred=y_pred_test,y_true=y_test)
print("Recall:",100*recall_score(y_pred=y_pred_test,y_true=y_test))
print("Precision:",100*precision_score(y_pred=y_pred_test,y_true=y_test))

print("Confusion Matrix")
print(CMatrix(CM))
print("-------------------------------------------------------------------------------")

#print(default)
#Making Individual Prediction'''

from sklearn.preprocessing import RobustScaler
robust_scaler=RobustScaler()
def make_ind_prediction(new_data):
    data=new_data.values.reshape(1,-1)
    data=robust_scaler.fit_transform(data)
    prob=logistic_regression.predict_proba(data)[0][1]
    if prob>=0.2:
        return 'will Not-Place'
    else:
        return 'Will Place'


pay=default[default['default']==0]
pay.head()



from collections import OrderedDict


new_customer=OrderedDict([
                          ('sex',0),('rural/urban',0),('cgpa',7.231409868),('Interest',0)
                          ])

new_customer=pd.Series(new_customer)
print("----------------------------------------------------------------")
print("Prediction of Input data")
print(make_ind_prediction(new_customer))










