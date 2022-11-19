#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Sat Nov  5 14:20:11 2022

@author: Gaëtan LE FLOCH
'''
#%% Import all packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from statsmodels.discrete.discrete_model import Logit
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection,tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
import dtreeviz.trees

#%%

df = pd.read_csv('data/metabolic_syndrome.csv')

df['MetabolicSyndrome'].value_counts(normalize=True) # 712 have metabolic syndrome
#%%
# ----------- Code to replicate the descriptive statistics --------------

(incomeBucket,ageBucket,WCBucket,
 BMIBucket,UABucket,URALBCRBucket,BGBucket,TBucket,
 HDLBucket) = ([],[],[],[],[],[],[],[],[])

for obs in df['Income']:
    if obs < 2000:
        incomeBucket.append('< 2000')
    elif obs >= 2000 and obs < 5000:
        incomeBucket.append('>= 2000 & < 5000')
    elif obs >= 5000 and obs < 7000:
        incomeBucket.append('>= 5000 and < 7000')
    else:
        incomeBucket.append('>= 7000')

for obs in df['Age']:
    if obs > 20 and obs <=35:
        ageBucket.append('20-35')
    elif obs > 35 and obs <= 60:
        ageBucket.append('36-60')
    else:
        ageBucket.append('61-80')
        
for obs in df['WaistCirc']:
    if obs > 63 and obs <= 100:
        WCBucket.append('63-100')
    else:
        WCBucket.append('> 100')
        
for obs in df['BMI']:
    if obs <= 18.5:
        BMIBucket.append('Underweight')
    elif obs > 18.5 and obs < 25:
        BMIBucket.append('Healthy')
    elif obs > 24.9 and obs < 30:
        BMIBucket.append('Overweight')
    else:
        BMIBucket.append('Obese')
        
for obs in df['UricAcid']:
    if obs < 7:
        UABucket.append('< 7')
    else:
        UABucket.append('> 7')
        
for obs in df['UrAlbCr']:
    if obs < 300:
        URALBCRBucket.append('< 300')
    else:
        URALBCRBucket.append('> 300')
        
for obs in df['BloodGlucose']:
    if obs < 54:
        BGBucket.append('< 54')
    else:
        BGBucket.append('> 54')
    
for obs in df['Triglycerides']:
    if obs < 200:
        TBucket.append('< 200')
    else:
        TBucket.append('> 200')
        
for obs in df['HDL']:
    if obs < 40:
        HDLBucket.append('< 40')
    else:
        HDLBucket.append('> 40')
        
        
dfDataViz = df.copy()
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
    
dfDataViz = dfDataViz.loc[df['MetabolicSyndrome'] == 'MetSyn']

for var in ['Sex','incomeBucket','AgeBucket','Race','Marital',
            'MetabolicSyndrome','WCBucket','Albuminuria','BMIBucket',
            'UABucket','URALBCRBucket','BGBucket','TBucket',
            'HDLBucket']: # Data for Table 1
    print(dfDataViz[var].value_counts())
    print(dfDataViz[var].value_counts(normalize=True))
#%%
# ------------ Chart of distributions among variables ------------------

fig, axs = plt.subplots(7,2)
i = 0
fig.suptitle('Distribution of biological variables')
fig.set_size_inches(18.5, 15.5)
plt.subplots_adjust(wspace=0.5,hspace=1)
for var in df.columns:
    row = i%7
    if i >= 7:
        column = 1
    else:
        column = 0
        
    if var in ['Age','WaistCirc','BMI','Albuminuria','UrAlbCr',
                'UricAcid','BloodGlucose','HDL','Triglycerides']:
        sns.kdeplot(df[var],ax=axs[row,column])
    else:
        sns.histplot(df,x=var,stat='density',ax=axs[row,column])
    axs[row,column].set_title(f'{var}')
    i+= 1
fig.show()
fig.savefig('firstDistribution.png', dpi=100)

fig, axs = plt.subplots(3,3)
i = 0
fig.suptitle('Distribution of biological variables')
fig.set_size_inches(18.5, 15.5)
plt.subplots_adjust(wspace=0.5,hspace=1)
for var in ['Age','WaistCirc','BMI','Albuminuria','UrAlbCr',
            'UricAcid','BloodGlucose','HDL','Triglycerides']:
    row = i%3
    if i < 3:
        column = 0
    elif i in [3,4,5]:
        column = 1
    else:
        column = 2
    sns.kdeplot(df[var],hue=df['Sex'],ax=axs[row,column])
    i+= 1
fig.show()
fig.savefig('secondDistribution.png', dpi=100)

sns.heatmap(df[['Sex','Age','WaistCirc','BMI','Albuminuria','UrAlbCr',
            'UricAcid','BloodGlucose','HDL','Triglycerides','Sex','MetabolicSyndrome']].corr())
plt.title('Correlations heatmap')
plt.show()
# -------------------------------------------------------------------
#%%
# One hot encode all categorical variables

df['MetabolicSyndrome'] = df['MetabolicSyndrome'].map({'MetSyn':1,'No MetSyn':0})
df['Sex'] = df['Sex'].map({'Male':1,'Female':0})
df = pd.concat([df,pd.get_dummies(df['Marital'],prefix='Marital_')],axis=1)
df = pd.concat([df,pd.get_dummies(df['Race'],prefix='Race_')],axis=1,)
df = df.drop(['Marital','Race'],axis=1)
# PCA

X = StandardScaler().fit_transform(df)

myPCA = PCA(n_components=2)
myPCA.fit_transform(X)

firstComponent,secondComponent = myPCA.components_[0],myPCA.components_[1]

for i, varnames in enumerate(df.columns):
    plt.scatter(firstComponent[i], secondComponent[i])
    plt.text(firstComponent[i], secondComponent[i], varnames)
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
fig.savefig('PCAResults.png', dpi=100)
    
# Not very informative
# Yet the waist circle and BMI (highly correlated between them) seem to be big determinants
#%%
# Logit regression
def doRegression(predictors):
    '''
    This function does the logit regression and outputs the summary.
    '''
    y = df['MetabolicSyndrome']
    X = df[predictors]
    myLogit = Logit(y,X)
    results = myLogit.fit()
    return results.summary()


predictors = df.columns.tolist()
predictors.remove('MetabolicSyndrome')
doRegression(predictors) # Regression no. 1

for i in ['Income','Marital__Divorced','Marital__Married',
                   'Marital__Separated','Marital__Single',
                   'Marital__Widowed']:
    predictors.remove(i)
doRegression(predictors) # Regression no. 2

for i in ['Race__Asian','Race__Black','Race__Hispanic','Race__MexAmerican',
          'Race__Other','Race__White']:
    predictors.remove(i)
doRegression(predictors) # Regression no. 3

predictors = df.columns.tolist()
predictors.remove('MetabolicSyndrome')
for i in ['Race__Asian','Race__Black','Race__Hispanic','Race__MexAmerican',
          'Race__Other','Race__White']:
    predictors.remove(i)
doRegression(predictors) # Regression no. 4

#%%
# Decision tree

#Define random_state constant for reproducability
rs = 42

#%%Load dataset and pre-process the data
metabolic = pd.read_csv('data/metabolic_syndrome.csv')

#Check for missing variables
check_na = metabolic.isnull()
check_na.isin([False]).any()

#Encoding of categorical variables
def encoding(df, col_name):
    '''Function takes the dataframe, the column name that is encoded and outputs the dataframe with the encoded catgecorical variables'''
    class_name = df[col_name].unique()
    df[col_name] = pd.Categorical(df[col_name], categories = class_name).codes
    return df

encoding(metabolic, 'Sex')
encoding(metabolic, 'Marital')
encoding(metabolic, 'Race')
encoding(metabolic, 'MetabolicSyndrome')

#Retrieve descriptive statistics
metabolic.describe().T

#%%Prepare dataset for machine learning classification

#Define features and target variable
X = metabolic.drop('MetabolicSyndrome', axis = 1) # define the feature variables
Y = metabolic['MetabolicSyndrome']# define thtarget variable

#Split in train and test set
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.30, random_state = rs, stratify = Y)

#%%Set up the classification algorithm
dtc = DecisionTreeClassifier(random_state=rs)

#%%Set grid search for hyperparameter tuning

#Identify the correct grid range for the alpha parameter
dtc = DecisionTreeClassifier(random_state=rs).fit(X_train,Y_train)
dtc_pruning = dtc.cost_complexity_pruning_path(X_train, Y_train)

pruning_summary = pd.DataFrame(dtc_pruning)
print(pruning_summary.head())

#Visualize
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
plt.title('Accuracy vs alphas for pruned trees')
plt.xlabel('ccp_alphas')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


#%%Set grid search for hyperparameter tuning
dtc_para = {'min_samples_split': range(2, 200, 10),
'min_samples_leaf': range(1,200,5),
'max_depth': range(1, 200,5), 
'ccp_alpha' : (0, 0.01, 0.02, 0.03, 0.04)}

#Deploy cross validation and fit the tree
cv_folds = StratifiedKFold(5, shuffle=True, random_state=rs)
dtc_cv = GridSearchCV(dtc, dtc_para, cv=cv_folds, n_jobs=-1)
dtc_cv.fit(X_train, Y_train) 


#%%Visualize test score distributions
results = pd.DataFrame(dtc_cv.cv_results_)
col_names = ['split0_test_score', 'split1_test_score', 'split2_test_score',
       'split3_test_score', 'split4_test_score', 'mean_test_score']

# for col in col_names:
#    sns.kdeplot(results[col], shade=True)

#Retrieve the best parameter and the accuracy
print(' Results from Grid Search ' )
print('\n The best estimator across ALL searched params:\n',dtc_cv.best_estimator_)
print('\n The best score across ALL searched params:\n',dtc_cv.best_score_)
print('\n The best parameters across ALL searched params:\n',dtc_cv.best_params_)


#%%
#Predict the the target using the best parameters chosen by the model 
y_pred = dtc_cv.best_estimator_.predict(X_test)

#Compute classification matrix
print(confusion_matrix(y_pred, Y_test))

#Viusalize the accuracy of the decision tree against the parameters of the grid search
sns.lineplot(data=results, x='param_min_samples_split', y='mean_test_score')
plt.show()

sns.lineplot(data=results, x='param_max_depth', y='mean_test_score')
plt.show()

sns.lineplot(data=results, x='param_min_samples_leaf', y='mean_test_score')
plt.show()

#%%Visualize best decision tree classifier
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


# %%
#Visualize the prediction path
# pick random X observation for demo

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
