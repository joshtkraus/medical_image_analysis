#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# clear variables
get_ipython().run_line_magic('reset', '-f')

# libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
from IPython.display import display
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

# Read Data
data = np.load('array_data/data_64.npy',allow_pickle=True)
labels = np.load('array_data/labels.npy',allow_pickle=True)

# Scale Data
scale_data = StandardScaler().fit_transform(data)

# Delete
del data


# In[ ]:


# Parameter Tuning

# ANN
def ann(x, y, test):
    print('Fitting ANN...')
    ## create model
    ann = MLPClassifier(max_iter=1000,
                        random_state=1)
    ## parameters to tune
    param_grid = {'hidden_layer_sizes':[(10,),(50,),(100,),(200,)],
                  'activation':['tanh', 'relu'],
                  'alpha':[0.001, 0.01, 0.1]
                 }
    ## use random search to tune parameters
    ann_gscv = RandomizedSearchCV(ann, param_grid, cv=5)
    ## fit model to data
    ann_gscv.fit(x, y)
    print(ann_gscv.best_params_)
    ## predict
    ann_pred = ann_gscv.predict(test)
    return ann_pred

# KNN
def knn(x, y, test):
    print('Fitting KNN...')
    ## create model
    knn = KNeighborsClassifier()
    ## parameters to tune
    param_grid = {'n_neighbors': np.arange(50, 200),
                 'weights':['uniform','distance'],
                 'p':[1,2]}
    ## use random search to tune parameters
    knn_gscv = RandomizedSearchCV(knn, param_grid, cv=5)
    ## fit model to data
    knn_gscv.fit(x, y)
    print(knn_gscv.best_params_)
    ## predict
    knn_pred = knn_gscv.predict(test)
    return knn_pred

# SVM
def svm(x, y, test):
    print('Fitting SVM...')
    ## create model
    svm = SVC(random_state=1)
    ## parameters to tune
    param_grid = {'C':[0.01,0.1,1,10],
                  'kernel':['poly','rbf'],
                  'decision_function_shape':['ovr'],
                  'shrinking':[True,False]
                 }
    # use random search to tune parameters
    svm_gscv = RandomizedSearchCV(svm, param_grid, cv=5)
    # fit model to data
    svm_gscv.fit(x, y)
    print(svm_gscv.best_params_)
    ## predict
    svm_pred = svm_gscv.predict(test)
    return svm_pred

# Logistic Regression
def log_reg(x, y, test):
    print('Fitting Logistic Regression...')
    ## create model
    log_reg = LogisticRegression(random_state=1)
    ## parameters to tune
    param_grid = {'penalty':['l1','l2','elasticnet'],
                  'C':[0.01,0.1,1],
                  'solver':['sag', 'saga']
                 }
    ## use random search to tune parameters
    log_reg_gscv = RandomizedSearchCV(log_reg, param_grid, cv=5)
    ## fit model to data
    log_reg_gscv.fit(x, y)
    print(log_reg_gscv.best_params_)
    ## predict
    log_pred = log_reg_gscv.predict(test)
    return log_pred

# Naive Bayes
def bayes(x, y, test):
    print('Fitting Naive Bayes...')
    ## create model
    bayes = GaussianNB()
    ## parameters to tune
    param_grid = {'var_smoothing':[1e-9,1e-7,1e-5,1e-3,1e-1,1e0]
                 }
    ## use random search to tune parameters
    bayes_gscv = RandomizedSearchCV(bayes, param_grid, cv=5)
    ## fit model to data
    bayes_gscv.fit(x, y)
    print(bayes_gscv.best_params_)
    ## predict
    bayes_pred = bayes_gscv.predict(test)
    return bayes_pred

# Gradient Boosting
def grad_boost(x, y, test):
    print('Fitting Gradient Boosting...')
    ## create model
    grad = AdaBoostClassifier(random_state=1)
    ## parameters to tune
    param_grid = {'n_estimators':[100,200,300,400,500],
                  'learning_rate':[0.01,0.1,1],
                 }
    ## use random search to tune parameters
    grad_gscv = RandomizedSearchCV(grad, param_grid, cv=5)
    ## fit model to data
    grad_gscv.fit(x, y)
    print(grad_gscv.best_params_)
    ## predict
    grad_pred = grad_gscv.predict(test)
    return grad_pred

# K-Means
def kmeans(x,y):
    print('Fitting K-Means...')
    ## create model
    kmeans = KMeans(random_state=1)
    ## parameters to tune
    param_grid = {'algorithm':['full','elkan'],
                  'n_init':[10,20,30,40],
                  'n_clusters':[2]
                 }
    ## use random search to tune parameters
    kmeans_gscv = RandomizedSearchCV(kmeans, param_grid, cv=5)
    ## fit model to data
    kmeans_gscv.fit(x)
    print(kmeans_gscv.best_params_)

# Performance
def perf(pred, real):
    print('Confusion Matrix')
    # confusion matrix
    conf_mat = confusion_matrix(real,pred)
    display(pd.DataFrame(conf_mat))
    print('Classification Report')
    # classificaiton report
    class_rep = classification_report(real,pred,output_dict=True)
    display(pd.DataFrame(class_rep))

# Full Implementation
def run(model,x_train,y_train,x_test,y_test):
    # Fit and Predict
    pred = model(x_train, y_train, x_test)
    # Examine Performance
    perf(y_test, pred)


# In[ ]:


# No Finding vs. Effusion, Balanced

# Subset Infiltration & No Finding
## subset
data_sub = scale_data[:,np.where((labels=='No Finding')|(labels=='Effusion'))[0]]
label_sub = labels[np.where((labels=='No Finding')|(labels=='Effusion'))[0]]
## convert response to binary
label_bin = np.where(label_sub == 'No Finding', 0, 1)

# Delete
del label_sub, scale_data, labels

# Balance (Undersampliung)
## separate
no_finding = np.where(label_bin==0)[0]
finding = np.where(label_bin==1)[0]
## set seed
np.random.seed(1)
## under sample
no_finding = np.random.choice(no_finding,
                              len(finding),
                              replace=False)
## indicies
ind = np.append(no_finding,finding)
## balance
data_small = data_sub[:,ind]
label_small = label_bin[ind]

# Delete
del no_finding, finding, ind, data_sub, label_bin

# Split Data
(x_train,x_test,
y_train,y_test) = train_test_split(data_small.T,label_small, 
                                       test_size=0.2, random_state=1,
                                       shuffle=True,stratify=label_small)

# Delete
del data_small, label_small

# Non-PCA Fitting (higher accuracy, slower computation)
print('Non-PCA Data')
# Tune Parameters
# ANN
run(ann,x_train,y_train,x_test,y_test)
# KNN
run(knn,x_train,y_train,x_test,y_test)
# SVM
run(svm,x_train,y_train,x_test,y_test)
# Logistic Regression
run(log_reg,x_train,y_train,x_test,y_test)
# Bayes
run(bayes,x_train,y_train,x_test,y_test)
# Gradient Boosting
run(grad_boost,x_train,y_train,x_test,y_test)
# K-Means
kmeans(x_test,y_test)

# Delete
del ann_pred,knn_pred,svm_pred,log_pred,bayes_pred,grad_pred,kmeans_pred


# In[ ]:


# PCA Fitting (lower accuracy, faster computation)
print('PCA Data')
# PCA
x_train_pca, x_test_pca = pca(x_train, x_test, y_train)

# Delete
del x_train, x_test

# Tune Parameters
# ANN
run(ann,x_train_pca,y_train,x_test_pca,y_test)
# KNN
run(knn,x_train_pca,y_train,x_test_pca,y_test)
# SVM
run(svm,x_train_pca,y_train,x_test_pca,y_test)
# Logistic Regression
run(log_reg,x_train_pca,y_train,x_test_pca,y_test)
# Bayes
run(bayes,x_train_pca,y_train,x_test_pca,y_test)
# Gradient Boosting
run(grad_boost,x_train_pca,y_train,x_test_pca,y_test)
# K-Means
kmeans(x_test_pca,y_test)

# Delete
del y_train, y_test, x_train_pca, x_test_pca

