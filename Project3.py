import numpy as np
import pandas as pd  # read and wrangle dataframes
from sklearn.cluster import KMeans
from sklearn import preprocessing
import matplotlib.pyplot as plt # visualization
import seaborn as sns # statistical visualizations and aesthetics
from sklearn.decomposition import PCA # dimensionality reduction
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.stats import boxcox # data transform
from sklearn.model_selection import (train_test_split, KFold , StratifiedKFold,
                                     cross_val_score, GridSearchCV,
                                     learning_curve, validation_curve) # model selection modules
from sklearn.base import BaseEstimator, TransformerMixin # To create a box-cox transformation class
from collections import Counter
import warnings
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
# load models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
#from xgboost import (XGBClassifier, plot_importance)
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from time import time
from sklearn.metrics import accuracy_score


df_test = pd.read_csv('test.csv')
df_train=pd.read_csv('train.csv')
#print(df_test.info(), "\n", df_train.info())
le=preprocessing.LabelEncoder()
df_train["species"]=le.fit_transform(df_train["species"])

df_train=df_train.drop(["id"], axis=1)
print(df_train)

X=df_train.iloc[:,1:193]
Y=df_train["species"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=10)

#Support Vector
svc_model=SVC()
svc_model.fit(X_train, y_train)
y_predict=svc_model.predict(X_test)
print('SVC Accuracy Score=%8.2f ' % (accuracy_score(y_test,y_predict)))

#print("\n", confusion_matrix(y_test,y_predict))

#Random Forest
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=100,n_jobs=-1)
clf_rf=rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
print ("Random Forest Accuracy= %.2f"% accuracy_score(y_test,y_pred))


#Naive
from sklearn.naive_bayes import BernoulliNB
nb=BernoulliNB()
clf_nb=nb.fit(X_train,y_train)
y_pred=nb.predict(X_test)
print ("Bernoulli Accuracy= %.2f" %accuracy_score(y_test,y_pred))


#Decision Tree
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred=dt.predict(X_test)
print("Decision Tree accuracy= %.2f" % accuracy_score(y_test,y_pred))

print("\n\n The best Classification method is Random Forest")

df_test=df_test.drop(["id"], axis=1)
X_test1=df_test
rf=RandomForestClassifier(n_estimators=100,n_jobs=-1)
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test1)
d_pred=le.inverse_transform(y_pred)
#print(d_pred)
df_test["species"]=d_pred
print(df_test)