#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
project_ds_metabolic.py: This file analyses historical data on patients with metabolic syndrome. The input dataset contains 2009 observations with 14 features. We output the following items: a) descriptive statistics, b) distribution tables, c)correlation matrix, d) logit regression, e) a decision tree classification algorithm.
'''
__author__      = 'Julia Schmidt, Gaetan Le Floch'
__copyright__   = 'Copyright 2022, Paris'
__status___     = 'Production'
__date__        = '15-12-2022'

#%% Import all packages
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from statsmodels.discrete.discrete_model import Logit

from sklearn.preprocessing import StandardScaler
from sklearn import model_selection,tree
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

import seaborn as sns
import matplotlib.pyplot as plt
from dtreeviz.trees import dtreeviz

import pickle

#%%Define random_state constant for reproducability
rs = 42

#%%Load and inspect the data
metabolic = pd.read_csv('data/metabolic_syndrome.csv')
print(metabolic.head(), '\n\n')

metabolic['MetabolicSyndrome'].value_counts(normalize=True)

#Check for missing variables
check_na = metabolic.isnull()
check_na.isin([False]).any()
print('There are no missing values in the dataset')

#Retrieve descriptive statistics
metabolic.describe().T

#%% Generate first descriptive statistics of the report (Table 1)
(incomeBucket,ageBucket,WCBucket,
 BMIBucket,UABucket,URALBCRBucket,BGBucket,TBucket,
 HDLBucket) = ([],[],[],[],[],[],[],[],[])

def buildBucket(df,var,bins):
    '''
    This function build buckets in order to draw descriptive statistics.
    '''
    buckets = []
    if len(bins) == 1:
        for obs in df[var]:
            if obs < bins[0]:
                buckets.append(f'< {bins[0]}')
            else:
                buckets.append(f'> {bins[0]}')
    else:
        for obs in df[var]:
            i = len(bins)
            while i != 0:
                if obs < bins[i-1]:
                    i -= 1
                else:
                    break
            if i == len(bins):
                buckets.append(f'>= {bins[i-1]}')
            elif i==0:
                buckets.append(f'< {bins[i]}')
            else:
                buckets.append(f'>= {bins[i-1]} and < {bins[i]}')
    return buckets

for obs in metabolic['Income']:
    if obs < 2000:
        incomeBucket.append('< 2000')
    elif obs >= 2000 and obs < 5000:
        incomeBucket.append('>= 2000 and < 5000')
    elif obs >= 5000 and obs < 7000:
        incomeBucket.append('>= 5000 and < 7000')
    else:
        incomeBucket.append('>= 7000')

for obs in metabolic['Age']:
    if obs > 20 and obs <=35:
        ageBucket.append('20-35')
    elif obs > 35 and obs <= 60:
        ageBucket.append('36-60')
    else:
        ageBucket.append('61-80')
        
for obs in metabolic['WaistCirc']:
    if obs > 63 and obs <= 94:
        WCBucket.append('63-94')
    else:
        WCBucket.append('> 94')
        
for obs in metabolic['BMI']:
    if obs <= 18.5:
        BMIBucket.append('Underweight')
    elif obs > 18.5 and obs < 25:
        BMIBucket.append('Healthy')
    elif obs > 24.9 and obs < 30:
        BMIBucket.append('Overweight')
    else:
        BMIBucket.append('Obese')
        
for obs in metabolic['UricAcid']:
    if obs < 7:
        UABucket.append('< 7')
    else:
        UABucket.append('> 7')
        
for obs in metabolic['UrAlbCr']:
    if obs < 300:
        URALBCRBucket.append('< 300')
    else:
        URALBCRBucket.append('> 300')
        
for obs in metabolic['BloodGlucose']:
    if obs < 200:
        BGBucket.append('< 200')
    else:
        BGBucket.append('> 200')
    
for obs in metabolic['Triglycerides']:
    if obs < 150:
        TBucket.append('< 150')
    else:
        TBucket.append('> 150')
        
for obs in metabolic['HDL']:
    if obs < 60:
        HDLBucket.append('< 60')
    else:
        HDLBucket.append('> 60')
        
        
dfDataViz = metabolic.copy()
(dfDataViz['incomeBucket'],dfDataViz['AgeBucket'],
 dfDataViz['WCBucket'],dfDataViz['BMIBucket'],
 dfDataViz['UABucket'],dfDataViz['URALBCRBucket'],
 dfDataViz['BGBucket'],dfDataViz['TBucket'],
 dfDataViz['HDLBucket']) = (incomeBucket,ageBucket,WCBucket,
                              BMIBucket, UABucket,URALBCRBucket,BGBucket,TBucket,
                              HDLBucket)

for var in ['Sex','incomeBucket','AgeBucket','Race','Marital',
            'MetabolicSyndrome','WCBucket','Albuminuria','BMIBucket',
            'UABucket','URALBCRBucket','BGBucket','TBucket',
            'HDLBucket']: # Data for Table 1
    print(dfDataViz[var].value_counts())
    print(dfDataViz[var].value_counts(normalize=True))


#%% Analysis of the distributions among variables (Figure 1)

#Densities of biological variables overall
fig, axs = plt.subplots(7,2)
i = 0
fig.suptitle('Density functions of biological variables')
fig.set_size_inches(18.5, 15.5)
plt.subplots_adjust(wspace=0.5,hspace=1)
for var in metabolic.columns:
    row = i%7
    if i >= 7:
        column = 1
    else:
        column = 0
        
    if var in ['Age','WaistCirc','BMI','Albuminuria','UrAlbCr',
                'UricAcid','BloodGlucose','HDL','Triglycerides']:
        sns.kdeplot(metabolic[var],ax=axs[row,column])
    else:
        sns.histplot(metabolic,x=var,stat='density',ax=axs[row,column])
    axs[row,column].set_title(f'{var}')
    i+= 1
fig.show()
fig.savefig('firstDistribution.png', dpi=100)


#Distribution of biological variables by sex
fig, axs = plt.subplots(4,2)
i = 0
fig.set_size_inches(18.5, 15.5)
fig.suptitle('Distribution of biological variables')
plt.subplots_adjust(wspace=0.5,hspace=1)
for var in ['WaistCirc','BMI','Albuminuria','UrAlbCr',
            'UricAcid','BloodGlucose','HDL','Triglycerides']:
    row = i%4
    if i < 4:
        column = 0
    else:
        column = 1
    sns.kdeplot(data=metabolic,x=var,hue='Sex',ax=axs[row,column])
    i+= 1
fig.show()
fig.savefig('secondDistribution.png', dpi=100)

#%% Correlation matrix (Figure 2)
heatmap = sns.heatmap(metabolic[['Sex','Age','WaistCirc','BMI','Albuminuria','UrAlbCr','UricAcid','BloodGlucose','HDL','Triglycerides','Sex','MetabolicSyndrome']].corr())
plt.show()
heatmap.figure.savefig("Heatmap.png",bbox_inches='tight')

#%% Logit regression 

#One hot encoding (required for logit regression)
metabolic['MetabolicSyndrome'] = metabolic['MetabolicSyndrome'].map({'MetSyn':1,'No MetSyn':0})
metabolic['Sex'] = metabolic['Sex'].map({'Male':1,'Female':0})
df = pd.concat([metabolic,pd.get_dummies(metabolic['Marital'],prefix='Marital_')],axis=1)
df = pd.concat([df,pd.get_dummies(df['Race'],prefix='Race_')],axis=1,)
df = df.drop(['Marital','Race'],axis=1)


def doRegression(predictors):
    '''
    This function takes as input the predictors of a logit regression and outputs a results summary.
    '''
    y = df['MetabolicSyndrome']
    X = df[predictors]
    myLogit = Logit(y,X)
    results = myLogit.fit()
    return results.summary()


predictors = df.columns.tolist()
predictors.remove('MetabolicSyndrome')
doRegression(predictors) # Regression no. 1

predictors = df.columns.tolist()
predictors.remove('MetabolicSyndrome')
for i in ['Marital__Divorced','Marital__Married',
                   'Marital__Separated','Marital__Single',
                   'Marital__Widowed']:
    predictors.remove(i)
doRegression(predictors) # Regression no. 2 (no marital)

predictors = df.columns.tolist()
predictors.remove('MetabolicSyndrome')
for i in ['Race__Asian','Race__Black','Race__Hispanic','Race__MexAmerican',
          'Race__Other','Race__White']:
    predictors.remove(i)
doRegression(predictors) # Regression no. 3 (no race)


predictors = df.columns.tolist()
predictors.remove('MetabolicSyndrome')
for i in ['Marital__Divorced','Marital__Married',
                   'Marital__Separated','Marital__Single',
                   'Marital__Widowed', 'Race__Asian','Race__Black','Race__Hispanic','Race__MexAmerican',
          'Race__Other','Race__White']:
    predictors.remove(i)
doRegression(predictors) # Regression no. 4 (no marital nor race)

#%% Implementation of a decision tree

# Encoding of categorical variables
def encoding(df, col_name):
    '''Function takes the dataframe, the column name that is encoded and outputs the dataframe with the encoded catgecorical variables'''
    class_name = df[col_name].unique()
    df[col_name] = pd.Categorical(df[col_name], categories = class_name).codes
    return df

encoding(metabolic, 'Sex')
encoding(metabolic, 'Marital')
encoding(metabolic, 'Race')
encoding(metabolic, 'MetabolicSyndrome')

#Define features and target variable
X = metabolic.drop('MetabolicSyndrome', axis = 1) # define the feature variables
Y = metabolic['MetabolicSyndrome']# define thtarget variable

#Split in train and test set
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.30, random_state = rs, stratify = Y)

#Set up the decisiontree classifier
dtc = DecisionTreeClassifier(random_state=rs)

#Set grid search for hyperparameter tuning
#Identify the correct grid range for the alpha parameter
dtc = DecisionTreeClassifier(random_state=rs).fit(X_train,Y_train)
dtc_pruning = dtc.cost_complexity_pruning_path(X_train, Y_train)
pruning_summary = pd.DataFrame(dtc_pruning)
print(pruning_summary.head())

#Visualize the pruning parameters against the impurity
plt.plot(dtc_pruning['ccp_alphas'], dtc_pruning['impurities'])
plt.title('Alpha vs. Impurities for a decision tree classifier')
plt.xlabel('Alphas')
plt.ylabel('Impurities')
plt.legend()
plt.show()

#Test which range of alpha parameters would be suitable for the grid search
accuracy_train = []
accuracy_test = []
for a in dtc_pruning['ccp_alphas']:
    dtc = DecisionTreeClassifier(random_state=rs, ccp_alpha = a).fit(X_train, Y_train)
    accuracy_train.append(accuracy_score(Y_train, dtc.predict(X_train)))
    accuracy_test.append(accuracy_score(Y_test, dtc.predict(X_test)))

alphas = dtc_pruning['ccp_alphas']
plt.plot(alphas, accuracy_train, label = 'Accuracy train')
plt.plot(alphas, accuracy_test, label = 'Accuracy test')
plt.xlabel('ccp_alphas')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Define grid search for hyperparameter tuning
dtc_para = {'min_samples_split': range(2, 200, 10),
'min_samples_leaf': range(1,200,5),
'max_depth': range(1, 200,5), 
'ccp_alpha' : (0, 0.01, 0.02, 0.03, 0.04)}

#Deploy stratified cross validation (imbalanced target) and fit the tree
cv_folds = StratifiedKFold(5, shuffle=True, random_state=rs) 
dtc_cv = GridSearchCV(dtc, dtc_para, cv=cv_folds, n_jobs=-1)
dtc_cv.fit(X_train, Y_train) 

#Visualize test score distributions
results = pd.DataFrame(dtc_cv.cv_results_)
col_names = ['split0_test_score', 'split1_test_score', 'split2_test_score',
       'split3_test_score', 'split4_test_score', 'mean_test_score']

#Retrieve the best parameter and their respective accuracies
print(' Results from Grid Search ' )
print('\n The best estimator across ALL searched params:\n',dtc_cv.best_estimator_)
print('\n The best score across ALL searched params:\n',dtc_cv.best_score_)
print('\n The best parameters across ALL searched params:\n',dtc_cv.best_params_)


#Predict the the target using the best parameters chosen by the model 
y_pred = dtc_cv.best_estimator_.predict(X_test)

#Evaluation matrix of the prediction
def evaluation_tree(Y_test, y_pred):
    accuracy = accuracy_score(Y_test, y_pred)
    recall = recall_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred)
    print(confusion_matrix(Y_test, y_pred))
    df = pd.DataFrame({'Metrics': 'best model', 'Accuracy': accuracy, 'recall': recall, 'precision': precision}, index=[0])
    return df

evaluation_tree(Y_test, y_pred)

#Visualize the accuracy of the decision tree against the parameters of the grid search

def plot_tree_param(data, x, y):
    '''Function takes the dataframe, and the x and y axis parameters and outputs a plot'''
    sns.lineplot(data = data, x = x, y = y)
    return plt.show()

#Plot the different parameters 
params = ['param_max_depth', 'param_min_samples_leaf', 'param_min_samples_split']
for i in params: 
    plot_tree_param(results, i, 'mean_test_score')

#Visualize best decision tree classifier
dtc = DecisionTreeClassifier(random_state=rs, max_depth = 6, min_samples_leaf=6, min_samples_split= 22)
dtc.fit(X_train, Y_train) 
tree.plot_tree(dtc)

fn=['Age','Sex','Marital','Income', 'Race', 'WaistCirc', 'BMI', 'Albuminuria', 'UrAlbCr', 'UricAcid', 'BloodGlucose', 'HDL', 'Triglyzerides']
cn=['MS no', 'MS yes']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(dtc,
               feature_names = fn, 
               class_names=cn,
               filled = True)
fig.savefig('decision_tree.png')

#Visualize the prediction path (pick random X observation for demonstration)
fig = plt.figure(figsize=(25,20))
dtc = DecisionTreeClassifier(random_state=rs, max_depth = 6, min_samples_leaf=6, min_samples_split= 22)
dtc.fit(X_train, Y_train) 

#pick random X for demo
datapoint = X_train.iloc[np.random.randint(0, len(X_train)),:].values

viz = dtreeviz(dtc,
               X_train,
               Y_train,
               target_name= 'Metabolic Syndrome',
               feature_names=X.columns,
               title='Metabolic syndrome classification',
               class_names=['0', '1'],
               scale=1.3, 
               X = datapoint)
viz
pickle.dump(dtc, open('dtc.sav', 'wb'))