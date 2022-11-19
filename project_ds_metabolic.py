#Import libraries
import numpy as np
import pandas as pd

#Import scikit learn
from sklearn import model_selection
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer
from sklearn import tree
from dtreeviz.trees import *
import graphviz

#Import visualization
import seaborn as sns
import matplotlib.pyplot as plt

#Define random_state constant for reproducability
rs = 42

#%%Load dataset and pre-process the data
metabolic = pd.read_csv('metabolic_syndrome.csv')
print(metabolic.head(), '\n\n')

#Check for missing variables
check_na = metabolic.isnull()
check_na.isin([False]).any()
print('There are no missing values in the dataset')

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
plt.title("Alpha vs. Impurities for a decision tree classifier")
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
plt.title("Accuracy vs alphas for pruned trees")
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
print(" Results from Grid Search " )
print("\n The best estimator across ALL searched params:\n",dtc_cv.best_estimator_)
print("\n The best score across ALL searched params:\n",dtc_cv.best_score_)
print("\n The best parameters across ALL searched params:\n",dtc_cv.best_params_)


#%%
#Predict the the target using the best parameters chosen by the model 
y_pred = dtc_cv.best_estimator_.predict(X_test)

#Compute classification matrix
print(confusion_matrix(y_pred, Y_test))

#Viusalize the accuracy of the decision tree against the parameters of the grid search
sns.lineplot(data=results, x="param_min_samples_split", y="mean_test_score")
plt.show()

sns.lineplot(data=results, x="param_max_depth", y="mean_test_score")
plt.show()

sns.lineplot(data=results, x="param_min_samples_leaf", y="mean_test_score")
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
               title="Metabolic syndrome classification",
               class_names=['0', '1'],
               scale=1.3, 
               X = datapoint)
viz

# %%
