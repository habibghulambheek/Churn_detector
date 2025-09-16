import os 
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score,precision_score,fbeta_score, make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost  import XGBClassifier
from imblearn.over_sampling import SMOTE
RANDOM_INT = 42 # For reproducible results
curr_dir = os.getcwd()
path = os.path.join(curr_dir, "Dataset.csv")
# f0_75 =  make_scorer(fbeta_score, beta = 0.75,pos_label  = 1)
df = pd.read_csv(path)
df = df.drop(columns=["customerID"])
discrete_Features  = list(df.columns)
discrete_Features.remove("tenure")
discrete_Features.remove("MonthlyCharges")
discrete_Features.remove("TotalCharges")
mean_tc = "2279.734304"
df["TotalCharges"] = df["TotalCharges"].replace(" ", mean_tc)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

df = pd.get_dummies(data=df, prefix= discrete_Features, columns=discrete_Features,drop_first=True)

Y = df["Churn_Yes"]
X = df.drop(columns=["Churn_Yes"])

# Splitting Data For Cross Validation and Gernalization

X_train, _X, y_train,_y = train_test_split(X,Y, test_size=0.6 , random_state= RANDOM_INT)
X_cv, X_test, y_cv, y_test = train_test_split(_X,_y, test_size= 0.5, random_state= RANDOM_INT)

del _X,_y
# smote = SMOTE()
# X_train,y_train = smote.fit_resample(X_train,y_train)
neg_values  = np.sum(y_train.values ==0)
pos_values  = np.sum(y_train.values ==1)
ratio =  neg_values/pos_values

# ================================
# RANDOMIZED SEARCH CV OPTIMIZATION
# ================================
# Here, I have applied RandomizedSearchCV to explore a predefined grid of 
# hyperparameters. This method randomly samples parameter combinations from the grid. 
# ================================

# xgb = XGBClassifier(random_state = RANDOM_INT)
# param_grid = {
#     'n_estimators': [200,300,400,450,475, 500, 600],          
#     'max_depth': [3, 5,6,7,9],                   
#     'learning_rate': np.linspace(0.01,0.2,20),       
#     'subsample': [0.6,0.7,0.8, 1.0],                  
#     'colsample_bytree': [0.6,0.7,0.8, 1.0],          
#     'min_child_weight': [1,3,5,7,10,15,20, 25],       
#     'scale_pos_weight': [ratio*0.2,ratio*0.3,ratio*0.5,ratio*0.8,ratio, ratio * 1.5, ratio*2, ratio*3],
#     'gamma': [0, 0.1,0.2,0.3, 0.5, 1,2],
#     'reg_alpha': [0, 0.1, 0.5, 1,1.5,2,4,5,7,8],
#     'reg_lambda': [0.5, 1, 2,2.5,4,5,3,6,7]}


# grid_search =  RandomizedSearchCV(estimator=xgb,param_distributions=param_grid,n_iter = 100, verbose = 2, scoring = f0_75, cv=5,n_jobs=-1)
# grid_search.fit(X_train,y_train)
# optimal_xgb = grid_search.best_estimator_
# print("Best parameters:",grid_search.best_params_)

# ================================
# OPTUNA HYPERPARAMETER OPTIMIZATION
# ================================
# In this section, I have used Optuna to automatically search for the best set of 
# hyperparameters for the XGBoost model. Optuna explores the search space 
# efficiently using trial-based optimization and selects the parameters 
# that maximize the chosen evaluation metric (f0_75 here).
# ================================

# def objective(trial):
#     param_grid = {
#         'n_estimators': trial.suggest_int('n_estimators', 200, 600,step=50),
#         'max_depth': trial.suggest_int('max_depth', 3, 9, step=1),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 28, step=1),
#         'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 3),
#         'gamma': trial.suggest_float('gamma', 0, 2),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0, 8.0),
#         'reg_lambda': trial.suggest_float('reg_lambda', 0, 8.0)
#     }

#     xgb = XGBClassifier(**param_grid, random_state=RANDOM_INT, eval_metric='logloss')
    
#     score = cross_val_score(xgb, X_train, y_train, scoring=f0_75, cv=5, n_jobs=-1)
#     return float(score.mean())

# study = optuna.create_study(direction='maximize')
# study.optimize(objective,n_trials=400, n_jobs=-1)
# print("Best parameters:",study.best_trial.params)
# XGB = XGBClassifier(**study.best_trial.params,random_state = RANDOM_INT)


# Optimal Parameters Identified
Best_parameters =  {'n_estimators': 550, 'max_depth': 4, 'learning_rate': 0.010392273241971246, 'subsample': 0.9393910128310066, 'colsample_bytree': 0.8531888476153595, 'min_child_weight': 4, 'scale_pos_weight': 1.7554071935290292, 'gamma': 0.9499349470683777, 'reg_alpha': 6.81875710990829, 'reg_lambda': 4.625309596379262}
xgb = XGBClassifier(**Best_parameters,random_state = RANDOM_INT)
xgb.fit(X_train,y_train)
y_hat_train =  (xgb.predict_proba(X_train)[:,1] >= 0.5).astype('int')
y_hat_cv =  (xgb.predict_proba(X_cv)[:,1] >= 0.5).astype('int')
y_hat_test =  (xgb.predict_proba(X_test)[:,1]  >= 0.5).astype('int')

train_error = np.round(np.mean((y_hat_train != y_train.values).astype('int'))*100,2)
cv_error =    np.round(np.mean((y_hat_cv != y_cv.values).astype('int'))*100,2)
test_error =  np.round(np.mean((y_hat_test !=y_test.values).astype('int'))*100,2)



print('-' * 30)
print('XGB Classifier')
print('-' * 30)
print('Training')
print('-' * 30)
print(f"Training error: {train_error}%")
print(f"Accuracy:\t{accuracy_score( y_train,y_hat_train)*100:.2f} %")
print(f"Recall:\t\t{recall_score( y_train,y_hat_train)*100:.2f} %")
print(f"percision:\t{precision_score( y_train,y_hat_train)*100:.2f} %")
print(f"F1 score:\t{f1_score( y_train,y_hat_train)*100:.2f} %")
confusion_mat = confusion_matrix( y_train,y_hat_train)
print("confusion matrix:")
print("----------------")
print("Actual Value\tNo\tYes")
print(f"No\t\t{confusion_mat[0][0]}\t{confusion_mat[0][1]}")
print(f"Yes\t\t{confusion_mat[1][0]}\t{confusion_mat[1][1]}")
print('-' * 30)



print('-' * 30)
print('Cross Validation')
print('-' * 30)
print(f"CV error: {cv_error}%")
print(f"Accuracy:\t{accuracy_score( y_cv,y_hat_cv)*100:.2f} %")
print(f"Recall:\t\t{recall_score( y_cv,y_hat_cv)*100:.2f} %")
print(f"percision:\t{precision_score( y_cv,y_hat_cv)*100:.2f} %")
print(f"F1 score:\t{f1_score( y_cv,y_hat_cv)*100:.2f} %")
confusion_mat = confusion_matrix( y_cv,y_hat_cv)
print("confusion matrix:")
print("----------------")
print("Actual Value\tNo\tYes")
print(f"No\t\t{confusion_mat[0][0]}\t{confusion_mat[0][1]}")
print(f"Yes\t\t{confusion_mat[1][0]}\t{confusion_mat[1][1]}")
print('-' * 30)

print('-' * 30)
print('Testing')
print('-' * 30)
print(f"Test error: {test_error}%")
print(f"Accuracy:\t{accuracy_score( y_test,y_hat_test)*100:.2f} %")
print(f"Recall:\t\t{recall_score( y_test,y_hat_test)*100:.2f} %")
print(f"percision:\t{precision_score( y_test,y_hat_test)*100:.2f} %")
print(f"F1 score:\t{f1_score( y_test,y_hat_test)*100:.2f} %")
confusion_mat = confusion_matrix( y_test,y_hat_test)
print("confusion matrix:")
print("----------------")
print("Actual Value\tNo\tYes")
print(f"No\t\t{confusion_mat[0][0]}\t{confusion_mat[0][1]}")
print(f"Yes\t\t{confusion_mat[1][0]}\t{confusion_mat[1][1]}")
print('-' * 30)
xgb.save_model("xgb.json")