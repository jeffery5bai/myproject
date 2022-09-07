
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime as dt
import streamlit as st
pd.options.display.max_columns = None

"## A Random Project"
"- Data Set: Telco-Customer-Churn"
"- Owner: Jeffery Bai"
"---"


st.write("### Dataset Overview")
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
st.write(df.head(10))
f"The Shape of Dataset: {df.shape}"

# Preprocess
df['SeniorCitizen'] = df['SeniorCitizen'].astype('str')
temp = list()
for i in range(df.shape[0]):
    try:
        temp.append(float(df.at[i, 'TotalCharges']))
    except:
        print(df.at[i, 'TotalCharges'])
        temp.append(0)
df['TotalCharges'] = temp

# Chart
"### EDA"
num_list = df.select_dtypes(['float','int']).columns
cat_list = df.select_dtypes("object").columns
cat_list = cat_list[1:]
n_bins = 20
"Histogram to Overview the Numeric Data"
option_n = st.selectbox("", options=num_list, index=0)
"Pie Chart to Overview the Categorical Data"
option = st.selectbox("", options=cat_list, index=len(cat_list)-1)

data = df.groupby(by=option).count().reset_index(option).sort_values(by=option).iloc[:, 1].values
# st.write(data)
labels = sorted(df[option].unique().tolist())
# st.write(labels)


fig, axs = plt.subplots(1, 2, figsize =(8, 6))
# plt.subplots_adjust(left=0.1,
#                     bottom=0.1, 
#                     right=0.9, 
#                     top=0.9, 
#                     wspace=0.4, 
#                     hspace=0.4
#                     )
axs[0].hist(df[option_n], bins=n_bins)
axs[1].pie(data, labels=labels, autopct='%1.1f%%', startangle=90)
axs[1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

st.pyplot(fig)

"---"
# Churn Persona
"### Persona of Customer Churn"
churn_df = df[df['Churn'] == 'Yes'].reset_index(drop=True).copy()
male = round((churn_df[churn_df['gender'] == 'Male'].shape[0] / churn_df.shape[0]) * 100, 2)
medianTotalCharges  = round(churn_df['TotalCharges'].median()/1000, 2)
medianOfTenure = round(churn_df['tenure'].median())
# monthlyTotalRatio = round(((churn_df['MonthlyCharges'] / churn_df['TotalCharges']) * 100).mean(), 2)




col1, col2, col3 = st.columns(3)
col1.metric("Male", f"{male} %", "0 %")
col2.metric("Median of Total Charges", f"$ {medianTotalCharges} k", "0 %")
col3.metric("Median of Tenure", f"{medianOfTenure} months", "0 %")

# fig1, ax = plt.subplots(figsize =(8, 6))
# # a = plt.figure(figsize=(4,3))
# ax.hist(churn_df['TotalCharges'])
# st.pyplot(fig1)


"---"
"## Prediction Model for Customer Churn"
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
import random

# categorical feature preprocessing
le = LabelEncoder()
ml_table = df.copy()
for i in range(len(cat_list)):
    col_name = f"{cat_list[i]}_N"
    ml_table[col_name] = le.fit_transform(ml_table[cat_list[i]].values)
    ml_table = ml_table.drop(cat_list[i], axis=1)

# st.write(ml_table)

# active_df = ml_table[ml_table['Churn_N'] == 0].reset_index(drop=True).copy()
active_df = ml_table[ml_table['Churn_N'] == 0].copy()
holdout_list = random.sample(set(active_df.index), 200)
remain_list = list(set(ml_table.index) - set(holdout_list))
holdout = ml_table.filter(items = holdout_list, axis=0).reset_index(drop=True)
ml_table = ml_table.filter(items = remain_list, axis=0).reset_index(drop=True)


X = ml_table.iloc[:, 1:ml_table.shape[1]-1].copy()
Y = ml_table['Churn_N'].copy()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=10)


model = xgb.XGBClassifier(seed=10)
model.fit(X_train, Y_train)
pred_y = model.predict(X_test)
real_y = Y_test.values

"**Confusion Matrix of The Testing Set**"
fpr, tpr, thresholds = roc_curve(real_y, pred_y, pos_label=1)
f"**The AUC Score:  {round(auc(fpr, tpr), 3)} , Accuracy Score:  {round(accuracy_score(real_y, pred_y), 3)}  of the Testing Set**"
st.write(confusion_matrix(real_y, pred_y))

holdout_x = holdout.iloc[:, 1:holdout.shape[1]-1].copy()
holdout_y = holdout['Churn_N'].copy()
holdout_pred_y = model.predict(holdout_x)
holdout_real_y = holdout_y.values


# "**Confusion Matrix of The Holdout Set**"
# fpr, tpr, thresholds = roc_curve(holdout_real_y, holdout_pred_y, pos_label=1)
# # f"**The AUC Score of The Holdout Set: {round(auc(fpr, tpr), 3)}**"
# f"**The Accuracy Score of The Holdout Set: {round(accuracy_score(holdout_real_y, holdout_pred_y), 3)}**"
# st.write(confusion_matrix(holdout_real_y, holdout_pred_y))

"\n"
"### Namelist of Potential Churn"
pred_p = model.predict_proba(holdout_x)
pred_p = pd.DataFrame(pred_p, columns=['0', '1'])

holdout['Probabilities_loss(1)'] = round(pred_p['1'], 3).copy()
holdout['Probabilities_stay(0)'] = round(pred_p['0'], 3).copy()
holdout['Prediction'] = holdout_pred_y

namelist = holdout[['customerID', 'Probabilities_stay(0)', 'Probabilities_loss(1)', 'Prediction']]
namelist = namelist.sort_values(by='Probabilities_loss(1)', ascending=False).reset_index(drop=True)

def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')
csv_file = convert_df(namelist)
st.download_button(label="Download Result",
                   data=csv_file,
                   file_name="Namelist.csv"
                   )


first_p = st.slider('The First __% of Potential Churn Namelist', 0, 100, 10) / 100

st.write(namelist.head(round(namelist.shape[0] * first_p)))

