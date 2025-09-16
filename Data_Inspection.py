import os
import numpy as no
import pandas as pd
from matplotlib import pyplot
import seaborn as sns
pd.set_option("display.max_columns", None)
curr_dir = os.getcwd()
path = os.path.join(curr_dir, "Dataset.csv")
df = pd.read_csv(path)
print(df.columns)
df = df.drop(columns="customerID")
df["TotalCharges"] = df["TotalCharges"].replace(" ", '0')
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

# print(df.head(10))
# print(df.dtypes)
# print(df.info())
print(df.describe(include="all"))

# Exploring the reason why people churn
# churn clossification
ax = sns.countplot(data = df, x =  'Churn')
pyplot.title("Count plot for churn")
ax.bar_label(ax.containers[0])
pyplot.show()

# Churn w.r.t categorical data

col_names  = list(df.columns)
col_names.remove("tenure")
col_names.remove("MonthlyCharges")
col_names.remove("TotalCharges")
col_names.remove("Churn")
continuos_features = ["tenure", 'MonthlyCharges', "TotalCharges"]

churn_conv  = (df["Churn"].values == 'Yes')
for col in continuos_features:
    sns.histplot(data= df,x= col,hue= 'Churn', bins = 72)
    pyplot.show()

for i in range(len(col_names)):
    ax = sns.countplot(data = df, x = col_names[i], hue = 'Churn')
    pyplot.title(f"Churn by {col_names[i]}")
    for container in ax.containers:
        ax.bar_label(container)
    pyplot.show()

    ct = pd.crosstab(df[col_names[i]], df["Churn"],normalize="index") * 100
    _ax = ct.plot(kind='bar')
    for container in _ax.containers:
        _ax.bar_label(container, fmt="%.2f%%")
    pyplot.title(f"Churn percentage by {col_names[i]}")
    pyplot.ylabel("Percentage")
    pyplot.show()

