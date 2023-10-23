import pandas as pd
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization
from sklearn.metrics import make_scorer, accuracy_score


new = pd.read_csv('selected_variables.csv', index_col=0)
print('Selected features: ', new.columns)

xtrain = new.drop(['primary_close_flag', 'final_close_flag'],axis=1)
ytrain1 = new['primary_close_flag']
ytrain2 = new['final_close_flag']

del new

# Getting the best parameters

param_grid = {
    'n_estimators': (80, 120),
    'max_depth': (3, 8),
    'learning_rate': (0.1, 1.0),
}


acc_score = make_scorer(accuracy_score)


def opt_func1(n_estimators, max_depth, learning_rate):
    param_dict = {}
    param_dict['n_estimators'] = round(n_estimators)
    param_dict['max_depth'] = round(max_depth)
    param_dict['learning_rate'] = learning_rate
    xgbmodel1 = XGBClassifier(**param_dict)
    score = cross_val_score(xgbmodel1, xtrain, ytrain1, scoring=acc_score).mean()
    score = score.mean()
    return score


opt1 = BayesianOptimization(opt_func1, param_grid, random_state=110)
opt1.maximize(init_points=4, n_iter=2)

print(opt1.max['params'])
best_params1 = opt1.max['params']
# best paramters for first variate:
# {'learning_rate': 0.27985412907354357,'max_depth': 3.0949737478952617, 'n_estimators': 119.86096598118098}


def opt_func2(n_estimators, max_depth, learning_rate):
    param_dict = {}
    param_dict['n_estmators'] = n_estimators
    param_dict['max_depth'] = round(max_depth)
    param_dict['learning_rate'] = learning_rate
    xgbmodel1 = XGBClassifier(**param_dict)
    score = cross_val_score(xgbmodel1, xtrain, ytrain2, scoring=acc_score).mean()
    score = score.mean()
    return score


opt2 = BayesianOptimization(opt_func2, param_grid, random_state=110)
opt2.maximize(init_points=4, n_iter=3)

best_params2 = opt2.max['params']
print(best_params2)
# best parameters for the second variate:
# {'learning_rate': 0.2044529760745759, 'max_depth': 6.292276908730766, 'n_estimators': 95.02355931866175}

best_params1['max_depth'] = round(best_params1['max_depth'])
best_params1['n_estimators'] = round(best_params1['n_estimators'])
print(best_params1)
best_params2['max_depth'] = round(best_params2['max_depth'])
best_params2['n_estimators'] = round(best_params2['n_estimators'])
print(best_params2)

# Importing the test data
test = pd.read_csv('test.csv')
test.head()
test.columns

drop_cols = []
for i in test.columns:
    if i not in xtrain.columns:
        drop_cols.append(i)

test.drop(drop_cols,inplace=True,axis=1)

print('Train columns: ',len(xtrain.columns))
print("Test columns: ",len(test.columns))

# Making the predictions
xgb1 = XGBClassifier(**best_params1)
xgb1.fit(xtrain,ytrain1)
ypred1 = xgb1.predict(test)

xgb2 = XGBClassifier(**best_params2)
xgb2.fit(xtrain,ytrain2)
ypred2 = xgb2.predict(test)

# Creating the submissions file
sub = pd.read_csv('submission.csv')
sub['primary_close_flag'] = ypred1
sub['final_close_flag'] = ypred2

sub.to_csv('final_sub.csv',index=False)

