import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score,precision_score,fbeta_score, make_scorer
from sklearn.linear_model import LogisticRegression

RANDOM_INT = 42 # For reproducible results

curr_dir = os.getcwd()
path = os.path.join(curr_dir, "Dataset.csv")

df = pd.read_csv(path)
df = df.drop(columns=["customerID"])

discrete_Features  = list(df.columns)
discrete_Features.remove("tenure")
discrete_Features.remove("MonthlyCharges")
discrete_Features.remove("TotalCharges")
mean_total_charges = "2279.734304"
df["TotalCharges"] = df["TotalCharges"].replace(" ", mean_total_charges)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

df = pd.get_dummies(data=df, prefix= discrete_Features, columns=discrete_Features,drop_first=True)

Y = df["Churn_Yes"]
X = df.drop(columns=["Churn_Yes"])

# Splitting Data For Cross Validation and Gernalization

X_train, _X, y_train,_y = train_test_split(X,Y, test_size=0.6 , random_state= RANDOM_INT)
X_cv, X_test, y_cv, y_test = train_test_split(_X,_y, test_size= 0.5, random_state= RANDOM_INT)

del _X,_y


model = LogisticRegression(class_weight='balanced')
model.fit(X_train,y_train)



y_hat_train =  (model.predict_proba(X_train)[:,1] >= 0.5).astype('int')
y_hat_cv =  (model.predict_proba(X_cv)[:,1] >= 0.5).astype('int')
y_hat_test =  (model.predict_proba(X_test)[:,1]  >= 0.5).astype('int')

train_error = np.round(np.mean((y_hat_train != y_train.values).astype('int'))*100,2)
cv_error =    np.round(np.mean((y_hat_cv != y_cv.values).astype('int'))*100,2)
test_error =  np.round(np.mean((y_hat_test !=y_test.values).astype('int'))*100,2)



print('-' * 30)
print('Logistic Regression')
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
