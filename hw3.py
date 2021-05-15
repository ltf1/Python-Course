import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#reading data

df =pd.read_csv("cses4_cut.csv")
df.head()
df.shape
df['voted'].value_counts()

#changing name of the columns to better understand what they stand for

df_new = df.rename(columns={'D2002': 'GENDER', "D2003" : "EDUCATION", "D2004":"MARITAL_STATUS", "D2005":"UNION_MEMBERSHIP", "D2006":"UNION_MEMBERSHIP_OTHERS",
                            "D2007": "BUSINESS_EMPLOYERS_ASSOCIATION_MEMBERSHIP", "D2008":"FARMERS_ASSOCIATION_MEMBERSHIP", "D2009":"PROFESSIONAL_ASSOCIATION_MEMBERSHIP",
                            "D2010":"CURRENT_EMPLOYMENT_STATUS", "D2011":"MAIN_OCCUPATION", "D2012":"SOCIO_ECONOMIC_STATUS", "D2013":"EMPLOYMENT_TYPE",
                            "D2014":"INDUSTRIAL_SECTOR", "D2015":"SPOUSE_CURRENT_EMPLOYMENT_STATUS", "D2016":"SPOUSE_OCCUPATION", "D2017":"SPOUSE_SOCIO_ECONOMIC_STATUS",
                            "D2018":"SPOUSE_EMPLOYMENT_TYPE", "D2019":"SPOUSE_INDUSTRIAL_SECTOR", "D2020":"HOUSEHOLD_INCOME", "D2021":"NUMBER_IN_HOUSEHOLD_TOTAL",
                            "D2022":"NUMBER_CHILDREN_HOUSEHOLD_UNDER_18", "D2023":"NUMBER_HOUSEHOLD_UNDER_6", "D2024":"RELIGIOUS_SERVICES_ATTENDANCE",
                            "D2025":"RELIGIOSITY", "D2026":"RELIGIOUS_DENOMINATION", "D2027":"LANGUAGE_USUALLY_SPOKEN_HOME", "D2028":"REGION_OF_RESIDENCE",
                            "D2029":"RACE", "D2030":"ETHNICITY", "D2031":"RURAL_URBAN RESIDENCE", "age":"AGE", "voted":"VOTED"
})
df_new

#drop first column of the dataset
df_new = df_new.iloc[: , 1:]
df_new.head()
df_new.shape


# checking of High or Low Feature Variability

for col in df_new.columns:
    print(col, df_new[col].nunique(), len(df))


#Odd Values and Data Collection Mistakes
for col in df_new.columns:
    print(col, df_new[col].unique(), len(df))


#replacing missing values with Nan

df_new["EDUCATION"].replace({96: np.nan, 97: np.nan, 98: np.nan, 99: np.nan}, inplace=True)

df_new["MARITAL_STATUS"].replace({9: np.nan, 8: np.nan, 7: np.nan}, inplace=True)

df_new["UNION_MEMBERSHIP"].replace({9: np.nan, 8: np.nan, 7: np.nan}, inplace=True)

df_new["UNION_MEMBERSHIP_OTHERS"].replace({9: np.nan, 8: np.nan, 7: np.nan}, inplace=True)

df_new["BUSINESS_EMPLOYERS_ASSOCIATION_MEMBERSHIP"].replace({9: np.nan, 8: np.nan, 7: np.nan}, inplace=True)

df_new["FARMERS_ASSOCIATION_MEMBERSHIP"].replace({9: np.nan, 8: np.nan, 7: np.nan}, inplace=True)

df_new["PROFESSIONAL_ASSOCIATION_MEMBERSHIP"].replace({9: np.nan, 8: np.nan, 7: np.nan}, inplace=True)

df_new["CURRENT_EMPLOYMENT_STATUS"].replace({97: np.nan, 98: np.nan, 99: np.nan}, inplace=True)

df_new["MAIN_OCCUPATION"].replace({996: np.nan, 997: np.nan, 998: np.nan, 999: np.nan}, inplace=True)

df_new["SOCIO_ECONOMIC_STATUS"].replace({7: np.nan, 8: np.nan, 9: np.nan}, inplace = True)

df_new["EMPLOYMENT_TYPE"].replace({7: np.nan, 8: np.nan, 9: np.nan}, inplace=True)

df_new["INDUSTRIAL_SECTOR"].replace({7: np.nan, 8: np.nan, 9: np.nan}, inplace=True)

df_new["SPOUSE_CURRENT_EMPLOYMENT_STATUS"].replace({97:np.nan, 98:np.nan, 99:np.nan}, inplace=True)

df_new["SPOUSE_OCCUPATION"].replace({996:np.nan, 997:np.nan, 998:np.nan, 999:np.nan}, inplace=True)

df_new["SPOUSE_SOCIO_ECONOMIC_STATUS"].replace({7:np.nan, 8:np.nan, 9:np.nan}, inplace=True)

df_new["SPOUSE_EMPLOYMENT_TYPE"].replace({7:np.nan, 8:np.nan, 9:np.nan}, inplace=True)

df_new["SPOUSE_INDUSTRIAL_SECTOR"].replace({7:np.nan, 8:np.nan, 9:np.nan}, inplace=True)

df_new["HOUSEHOLD_INCOME"].replace({7:np.nan, 8:np.nan, 9:np.nan}, inplace=True)

df_new["NUMBER_IN_HOUSEHOLD_TOTAL"].replace({97:np.nan, 98:np.nan, 99:np.nan}, inplace=True)

df_new["NUMBER_CHILDREN_HOUSEHOLD_UNDER_18"].replace({97:np.nan, 98:np.nan, 99:np.nan}, inplace=True)

df_new["NUMBER_HOUSEHOLD_UNDER_6"].replace({97:np.nan, 98:np.nan, 99:np.nan}, inplace=True)

df_new["RELIGIOUS_SERVICES_ATTENDANCE"].replace({7:np.nan, 8:np.nan, 9:np.nan}, inplace=True)

df_new["RELIGIOSITY"].replace({7:np.nan, 8:np.nan, 9:np.nan}, inplace=True)

df_new["RELIGIOUS_DENOMINATION"].replace({97:np.nan, 98:np.nan, 99:np.nan}, inplace=True)

df_new["LANGUAGE_USUALLY_SPOKEN_HOME"].replace({997:np.nan, 998:np.nan, 999:np.nan}, inplace=True)

df_new["REGION_OF_RESIDENCE"].replace({99:np.nan}, inplace=True)

df_new["RACE"].replace({997:np.nan, 998:np.nan, 999:np.nan}, inplace=True)

df_new["ETHNICITY"].replace({997:np.nan, 998:np.nan, 999:np.nan}, inplace=True)

df_new["RURAL_URBAN RESIDENCE"].replace({7:np.nan, 8:np.nan, 9:np.nan}, inplace=True)

df_new.head()

df_new.isnull().sum().sum()

df_new.isnull().sum()


#visualizing the number of missing values across the variables
df_new.isnull().sum().plot.bar(figsize=(12,6))
plt.ylabel('Number of null values')
plt.xlabel('Variables')
plt.title('Nullness of the dataset')


df_new.drop('RELIGIOUS_DENOMINATION', inplace=True, axis=1) #drop this column because the data does not match with the codebook
df_new.drop('REGION_OF_RESIDENCE', inplace=True, axis=1) # I was not sure what to do with a 80 different values for this variable. So, I removed it
df_new.shape


# First I dropped columns whose half of the rows have a valid values.
#however, this has dramatically reduced the remaining dataset as there remanined less than one thousand observations.
df_new=df_new.dropna(how='any',axis=1,thresh=11000)  # then, I dropped clomuns that have less than 11000 valid values.
df_new


#dropping rows with the missing Values
df_new= df_new.dropna()
df.isnull().sum()
df_new.count()



sns.set(rc={'figure.figsize':(11.7,8.27)})
plt.subplot(231)
sns.countplot(x="GENDER", hue='VOTED', data=df_new)
plt.subplot(232)
sns.countplot(x="EDUCATION", hue='VOTED', data=df_new)
plt.subplot(233)
sns.countplot(x="MARITAL_STATUS", hue='VOTED', data=df_new)
plt.subplot(234)
sns.countplot(x="UNION_MEMBERSHIP", hue='VOTED', data=df_new)
plt.subplot(235)
sns.countplot(x="CURRENT_EMPLOYMENT_STATUS", hue='VOTED', data=df_new)
plt.subplot(236)
sns.countplot(x="RELIGIOUS_SERVICES_ATTENDANCE", hue='VOTED', data=df_new)

#correlation matrix
correlation_mat = df_new.corr()

sns.heatmap(correlation_mat,annot=True,linewidths=.5,cmap="YlGnBu")


#featuer enginerring - one-hot encoding
from sklearn.preprocessing import OneHotEncoder

copy_df_new=df_new.copy()

GEN_HOT = copy_df_new["GENDER"].values.reshape(-1,1)

onehot_encoder = OneHotEncoder(sparse=False)

GEN_OneHotEncoded = onehot_encoder.fit_transform(GEN_HOT)


copy_df_new["GEN1"] = GEN_OneHotEncoded[:,0]
copy_df_new["GEN2"] = GEN_OneHotEncoded[:,1]




EDU_HOT = copy_df_new["EDUCATION"].values.reshape(-1,1)

onehot_encoder = OneHotEncoder(sparse=False)

EDU_OneHotEncoded = onehot_encoder.fit_transform(EDU_HOT)


copy_df_new["EDU1"] = EDU_OneHotEncoded[:,0]
copy_df_new["EDU2"] = EDU_OneHotEncoded[:,1]
copy_df_new["EDU3"] = EDU_OneHotEncoded[:,2]
copy_df_new["EDU4"] = EDU_OneHotEncoded[:,3]
copy_df_new["EDU5"] = EDU_OneHotEncoded[:,4]
copy_df_new["EDU6"] = EDU_OneHotEncoded[:,5]
copy_df_new["EDU7"] = EDU_OneHotEncoded[:,6]
copy_df_new["EDU8"] = EDU_OneHotEncoded[:,7]
copy_df_new["EDU9"] = EDU_OneHotEncoded[:,8]



MARR_HOT = copy_df_new["MARITAL_STATUS"].values.reshape(-1,1)

onehot_encoder = OneHotEncoder(sparse=False)

MARR_OneHotEncoded = onehot_encoder.fit_transform(MARR_HOT)


copy_df_new["MARR1"] = MARR_OneHotEncoded[:,0]
copy_df_new["MARR2"] = MARR_OneHotEncoded[:,1]
copy_df_new["MARR3"] = MARR_OneHotEncoded[:,2]
copy_df_new["MARR4"] = MARR_OneHotEncoded[:,3]



UNM_HOT = copy_df_new["UNION_MEMBERSHIP"].values.reshape(-1,1)

onehot_encoder = OneHotEncoder(sparse=False)

UNM_OneHotEncoded = onehot_encoder.fit_transform(UNM_HOT)


copy_df_new["UNM1"] = UNM_OneHotEncoded[:,0]
copy_df_new["UNM2"] = UNM_OneHotEncoded[:,1]




EMP_HOT = copy_df_new["CURRENT_EMPLOYMENT_STATUS"].values.reshape(-1,1)

onehot_encoder = OneHotEncoder(sparse=False)

EMP_OneHotEncoded = onehot_encoder.fit_transform(EMP_HOT)


copy_df_new["EMP1"] = EMP_OneHotEncoded[:,0]
copy_df_new["EMP2"] = EMP_OneHotEncoded[:,1]
copy_df_new["EMP3"] = EMP_OneHotEncoded[:,2]
copy_df_new["EMP4"] = EMP_OneHotEncoded[:,3]
copy_df_new["EMP5"] = EMP_OneHotEncoded[:,4]
copy_df_new["EMP6"] = EMP_OneHotEncoded[:,5]
copy_df_new["EMP7"] = EMP_OneHotEncoded[:,6]
copy_df_new["EMP8"] = EMP_OneHotEncoded[:,7]
copy_df_new["EMP9"] = EMP_OneHotEncoded[:,8]
copy_df_new["EMP10"] = EMP_OneHotEncoded[:,9]

copy_df_new.head()



X = copy_df_new.drop(copy_df_new.columns[[0,1,2,3,4,7]],axis=1)
y =copy_df_new.VOTED
X.head()
y


#creating test and training data
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


#feature selection which features are most influencial
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition

feature_names = train_X.columns
## Anova
selector = feature_selection.SelectKBest(score_func=
               feature_selection.f_classif, k=10).fit(train_X, train_y)
anova_selected_features = feature_names[selector.get_support()]

## Lasso regularization
selector = feature_selection.SelectFromModel(estimator=
              linear_model.LogisticRegression(C=1, penalty="l1",
              solver='liblinear'), max_features=10).fit(train_X, train_y)
lasso_selected_features = feature_names[selector.get_support()]

## Plot
dtf_features = pd.DataFrame({"features":feature_names})
dtf_features["anova"] = dtf_features["features"].apply(lambda x: "anova" if x in anova_selected_features else "")
dtf_features["num1"] = dtf_features["features"].apply(lambda x: 1 if x in anova_selected_features else 0)
dtf_features["lasso"] = dtf_features["features"].apply(lambda x: "lasso" if x in lasso_selected_features else "")
dtf_features["num2"] = dtf_features["features"].apply(lambda x: 1 if x in lasso_selected_features else 0)
dtf_features["method"] = dtf_features[["anova","lasso"]].apply(lambda x: (x[0]+" "+x[1]).strip(), axis=1)
dtf_features["selection"] = dtf_features["num1"] + dtf_features["num2"]
sns.barplot(y="features", x="selection", hue="method", data=dtf_features.sort_values("selection", ascending=False), dodge=False)

## Importance
model = ensemble.RandomForestClassifier(n_estimators=100,
                      criterion="entropy", random_state=0)
model.fit(train_X, train_y)
importances = model.feature_importances_
## Put in a pandas dtf
dtf_importances = pd.DataFrame({"IMPORTANCE":importances,
            "VARIABLE":feature_names}).sort_values("IMPORTANCE",
            ascending=False)
dtf_importances['cumsum'] = dtf_importances['IMPORTANCE'].cumsum(axis=0)
dtf_importances = dtf_importances.set_index("VARIABLE")

## Plot
fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False)
fig.suptitle("Features Importance", fontsize=20)
ax[0].title.set_text('variables')

dtf_importances[["IMPORTANCE"]].sort_values(by="IMPORTANCE").plot(
                kind="barh", legend=False, ax=ax[0]).grid(axis="x")
ax[0].set(ylabel="")
ax[1].title.set_text('cumulative')
dtf_importances[["cumsum"]].plot(kind="line", linewidth=4,
                                 legend=False, ax=ax[1])
ax[1].set(xlabel="", xticks=np.arange(len(dtf_importances)),
          xticklabels=dtf_importances.index)
plt.xticks(rotation=70)
plt.grid(axis='both')
plt.show()


#selected features as a result of the Plot

X_names = ["AGE", "RELIGIOUS_SERVICES_ATTENDANCE", "MARR4", "EDU4", "MARR1", "GEN1",
"EDU3", "EMP1", "GEN2", "EDU2"]
X = copy_df_new[X_names]
y = copy_df_new.VOTED

X.shape


#creating test and training data ONCE AGAIN with the new variables
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

#applying machine learning model_selection

#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB # 1. choose model class
model = GaussianNB() # 2. instantiate model
model.fit(train_X, train_y) # 3. fit model to data
GaussianNB_preds = model.predict(val_X) # 4. predict on new data

from sklearn.metrics import accuracy_score
GaussianNB_score = accuracy_score(val_y, GaussianNB_preds)
GaussianNB_score

from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
print('Accuracy Score : ' + str(accuracy_score(val_y,GaussianNB_preds)))
print('Precision Score : ' + str(precision_score(val_y,GaussianNB_preds)))
print('Recall Score : ' + str(recall_score(val_y,GaussianNB_preds)))
print('F1 Score : ' + str(f1_score(val_y,GaussianNB_preds)))


from sklearn.metrics import confusion_matrix
print('Confusion Matrix : \n' + str(confusion_matrix(val_y,GaussianNB_preds)))


#LogisticRegression
import warnings

warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression().fit(train_X,train_y)
logr_pred = clf.predict(val_X)

from sklearn.metrics import accuracy_score
logr_score = accuracy_score(val_y, logr_pred)
print("Logistic Regression Accuracy Score:", logr_score)

#Logistic Regression Classifier Confusion matrix
from sklearn.metrics import confusion_matrix
print('Confusion Matrix : \n' + str(confusion_matrix(val_y,logr_pred)))


#optimize hyper parameters of LogisticRegression
from sklearn.model_selection import GridSearchCV
clf = LogisticRegression()
grid_values = {'penalty': ['l1', 'l2'],'C':[0.001,.009,0.01,.09,1,5,10,25]}
grid_clf_acc = GridSearchCV(clf, param_grid = grid_values,scoring = 'accuracy')
grid_clf_acc.fit(train_X,train_y)

print(grid_clf_acc.best_score_)
print(grid_clf_acc.best_params_)

#application of the best parameters to the data

model = grid_clf_acc.best_estimator_

Logistic_grid_preds = model.fit(train_X, train_y).predict(val_X)
Logistic_grid_score = accuracy_score(val_y, Logistic_grid_preds)
Logistic_grid_score


#decision tree model
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier() # 2. instantiate model
model.fit(train_X, train_y) # 3. fit model to data
Decision_preds = model.predict(val_X) # 4. predict on new data

from sklearn.metrics import accuracy_score
Decision_score = accuracy_score(val_y, Decision_preds)
print(f"decision tree score:", Decision_score)

#optimize hyper parameters of a DecisionTree

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

criterion= ["gini", "entropy"]
max_depth=[1,3,5,None]
splitter=["best", "random"]

grid = GridSearchCV(estimator=model, cv=3, param_grid=dict(criterion=criterion, max_depth=max_depth, splitter=splitter))

grid.fit(train_X, train_y)

print(grid.best_score_)
print(grid.best_params_)

# application of best parameters to the data
model = grid.best_estimator_

Decision_grid_preds = model.fit(train_X, train_y).predict(val_X)
Decision_grid_score = accuracy_score(val_y, Decision_grid_preds)
print(f"Decision Tree Score after tuning parameters:", Decision_grid_score)
