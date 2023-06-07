import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# File Infos
adultFileInfo = {
    "url": '/Users/kimminseok/C4.5/data/adult-edit.csv',
    "categorical_columns": ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'],
    "initial_drop_columns": [],
    "class_column": "income",
    "class_values": [">50K", "<=50K"]
}
# abaloneFileInfo = {
#     "url": '/Users/kimminseok/C4.5/data/abalone-edit.csv'
# }
bankFileInfo = {
    "url": '/Users/kimminseok/C4.5/data/bank.csv',
    "categorical_columns": ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome'],
    "initial_drop_columns": [],
    "class_column": "y",
    "class_values": ["yes", "no"]
}

#selected File
selectedFile = bankFileInfo

# Tree Model Info
criterion="entropy"
splitter="best"
max_depth=5
random_state=1

# Preprocess data
df = pd.read_csv(selectedFile["url"])

categorical_columns = selectedFile["categorical_columns"]
initial_drop_columns = selectedFile["initial_drop_columns"]
class_column = selectedFile["class_column"]

data = df.drop(labels=initial_drop_columns, axis=1)

X = data.drop(labels=class_column, axis=1)
y = data[class_column]

# 범주형 설명변수에 대한 Dummy data 생성
for c in categorical_columns:
    data.groupby(c)[c].count()

    X = X.drop(labels=c, axis=1)
    X_dummy = pd.get_dummies(data=data[c], prefix=c, drop_first=True)
    X = pd.concat(objs=[X,X_dummy], axis=1)

# Set Train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = DecisionTreeClassifier(
    criterion=criterion,
    splitter=splitter,
    max_depth=max_depth,
    random_state=random_state
).fit(X_train, y_train)

# Draw Tree Image
fig = plt.figure(figsize=(15, 10), facecolor='white')
plot_tree(clf, feature_names=X.columns, class_names=clf.classes_,)
plt.show()

## 변수 중요도
for i, col in enumerate(X.columns):
    print(f'{col} 중요도 : {clf.feature_importances_[i]}')

print(clf.get_params()) ## DecisionTreeClassifier 클래스 인자 설정 정보
print('정확도 : ', clf.score(X_test,y_test)) ## 성능 평가 점수(정확도 Accuracy)

predictions = clf.predict(X_test)

def class_column_encoder(val):
    return selectedFile['class_values'].index(val)

y_test_labels = list(map(class_column_encoder, y_test.values))
predictions_labels = list(map(class_column_encoder, predictions))

## Error rate
RMSE_tree = mean_squared_error(y_test_labels, predictions_labels)**0.5
print('RMSE : ', RMSE_tree)


