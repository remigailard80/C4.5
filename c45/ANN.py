''' 기본 모듈 및 시각화 모듈 '''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

''' 데이터 전처리 모듈 '''
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

''' Neural Network Regressor 모듈 '''
from sklearn.neural_network import MLPRegressor

''' 결과 평가용 모듈 '''
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

''' 기타 optional'''
import warnings, itertools

warnings.filterwarnings(action='ignore')
pd.set_option('display.max_columns', None)

## Setting Data
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

# filePath = '/Users/kimminseok/C4.5/data/toyotacorolla/ToyotaCorolla.csv'
# bank_marketing_data = '/Users/kimminseok/C4.5/data/bank-marketing/ToyotaCorolla.csv'
# categorical_columns = ['Fuel_Type']
# initial_drop_columns = ['Id', 'Model']
# class_column = 'Price'

selectedFile = bankFileInfo

# Preprocess Data
data = pd.read_csv(selectedFile['url'])
data = data.drop(labels=selectedFile['initial_drop_columns'], axis=1)

X = data.drop(labels=selectedFile['class_column'], axis=1)
y = data[selectedFile['class_column']]

# 범주형 설명변수에 대한 Dummy data 생성
for c in selectedFile['categorical_columns']:
    data.groupby(c)[c].count()

    X = X.drop(labels=c, axis=1)
    X_dummy = pd.get_dummies(data=data[c], prefix=c, drop_first=True)
    X = pd.concat(objs=[X,X_dummy], axis=1)


def class_column_encoder(val):
    return selectedFile['class_values'].index(val)

# preprocessing y
y = pd.DataFrame(list(map(class_column_encoder, y.values)))

# Split test data and training data
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)

# Standardization
scaler = StandardScaler()
scaler.fit(train_x)

train_x = scaler.transform(train_x)
test_x = scaler.transform(test_x)

# param_grid_random = {
#     'hidden_layer_sizes': [(18,), (18,18,), (18,18,18,)],
#     'activation': ['tanh','relu','logistic'],
#     'solver': ['sgd', 'adam'],
#     'learning_rate': ['constant','adaptive','invscaling'],
#     'alpha': [0.01, 0.05],
#     'max_iter': [1000, 2000, 5000],
#     'early_stopping': [False, True],
#     'warm_start': [False]
# }

# MLP Regressor - need to change alpha and momentum, hidden layer size.
reg_mlp = MLPRegressor(
    activation = 'relu', 
    alpha = 0.01, 
    batch_size = 16,
    max_iter = 1000,
    solver= 'adam', 
    momentum= 0.9,
    verbose = True, 
    random_state = 1,
    learning_rate = 'adaptive',
    early_stopping = True
)

# Train MLP Regressor
reg_mlp.fit(train_x, train_y)

# Check trained status
# plt.figure(figsize=(20,10))

# train_loss_values = reg_mlp.loss_curve_
# plt.plot(train_loss_values,label='Train Loss')

# plt.legend(fontsize=20)
# plt.title("Learning Curve of trained MLP Regressor", fontsize=18)
# plt.show()

### 학습된 MLP Regressor 결과 확인 및 성능 평가 : Testing Data
test_y_pred = reg_mlp.predict(test_x)

print('----- Test Result -----')

print("Testing MSE : {:.3f}".format(mean_squared_error(test_y, test_y_pred)))
print("Testing RMSE : {:.3f}".format(np.sqrt(mean_squared_error(test_y, test_y_pred))))
print("Testing MAE : {:.3f}".format(mean_absolute_error(test_y, test_y_pred)))
print("Testing MAPE : {:.3f}".format(mean_absolute_percentage_error(test_y, test_y_pred)))
print("Testing R2 : {:.3f}".format(r2_score(test_y, test_y_pred)))

print('----- ----- -----')

# 산점도 그래프
# fig_values = np.concatenate([test_y.squeeze(), test_y_pred.squeeze()])
# vmin = np.min(fig_values) * 0.95
# vmax = np.max(fig_values) * 1.05

# plt.figure(figsize=(8, 8))
# plt.title('Actual values vs. Predicted values (Testing Data)', size=18)
# plt.scatter(test_y, test_y_pred)
# plt.plot([vmin, vmax], [vmin, vmax], color='grey', linestyle='dashed')
# plt.xlabel('Actual', size=16)
# plt.ylabel('Predicted', size=16)
# plt.show()
