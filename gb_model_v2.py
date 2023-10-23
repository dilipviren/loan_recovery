import pandas as pd
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization
from sklearn.metrics import make_scorer, accuracy_score

# Importing and preparing the data
new = pd.read_csv('selected_variables.csv', index_col=0)
print('Selected features: ', new.columns)

xtrain = new.drop(['primary_close_flag', 'final_close_flag'], axis=1)
ytrain1 = new['primary_close_flag']
ytrain2 = new['final_close_flag']

del new

# Defining the ranges to search for optimal parameters
param_grid = {
    'n_estimators': (80, 120),
    'max_depth': (3, 8),
    'learning_rate': (0.1, 1.0),
}

# Defining the accuracy metric
acc_score = make_scorer(accuracy_score)


# Function for the first model parameter tuning


def opt_func1(n_estimators, max_depth, learning_rate):
    param_dict = {'n_estimators': round(n_estimators), 'max_depth': round(max_depth), 'learning_rate': learning_rate}
    xgbmodel1 = XGBClassifier(**param_dict)
    score = cross_val_score(xgbmodel1, xtrain, ytrain1, scoring=acc_score).mean()
    score = score.mean()
    return score


# Function for the second model parameter tuning


def opt_func2(n_estimators, max_depth, learning_rate):
    param_dict = {'n_estimators': round(n_estimators), 'max_depth': round(max_depth), 'learning_rate': learning_rate}
    xgbmodel2 = XGBClassifier(**param_dict)
    score = cross_val_score(xgbmodel2, xtrain, ytrain2, scoring=acc_score).mean()
    score = score.mean()
    return score


# Optimizing the first model's parameters
opt1 = BayesianOptimization(opt_func1, param_grid, random_state=110)
opt1.maximize(init_points=8, n_iter=4)
best_params1 = opt1.max['params']
print('The optimal parameters for the first model: ', best_params1)

# Optimizing the second model's parameters
opt2 = BayesianOptimization(opt_func2, param_grid, random_state=110)
opt2.maximize(init_points=8, n_iter=4)
best_params2 = opt2.max['params']
print('The optimal parameters for the second model: ', best_params2)


# Summary of the best parameters for each model
best_params1['max_depth'] = round(best_params1['max_depth'])
best_params1['n_estimators'] = round(best_params1['n_estimators'])
print('Best Parameters for first mode:', best_params1)

best_params2['max_depth'] = round(best_params2['max_depth'])
best_params2['n_estimators'] = round(best_params2['n_estimators'])
print('Best parameters for second model', best_params2)

# Importing the test data
test = pd.read_csv('test.csv')
test.head()
test.columns

drop_cols = []
for i in test.columns:
    if i not in xtrain.columns:
        drop_cols.append(i)

test.drop(drop_cols,inplace=True,axis=1)

print('Train columns: ', len(xtrain.columns))
print("Test columns: ", len(test.columns))

# Making the predictions
xgb1 = XGBClassifier(**best_params1)
xgb1.fit(xtrain, ytrain1)
ypred1 = xgb1.predict(test)

xgb2 = XGBClassifier(**best_params2)
xgb2.fit(xtrain, ytrain2)
ypred2 = xgb2.predict(test)

# Creating the submissions file
sub = pd.read_csv('submission.csv')
sub['primary_close_flag'] = ypred1
sub['final_close_flag'] = ypred2

sub.to_csv('final_sub_v2.csv', index=False)
