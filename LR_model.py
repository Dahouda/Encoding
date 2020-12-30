"""
Regression is a statistical process for estimating the relationship among variables,
often to make predictions about some outcome
"""
import joblib
import models
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

tr_features = models.all_X_train
tr_labels = models.y_train
te_features = models.all_X_test
te_labels = models.y_test

print("--------- Check Data Shapes--------")
print(tr_features.shape, tr_labels.shape, te_features.shape, te_labels.shape)
print(tr_features.head())
print(tr_labels.head())

# Logistic Regression with Hyperparameters tuning

def print_results(results):
    print('BEST PARAMETERS: {}\n'.format(results.best_params_))
    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 2), round(std * 2, 2), params))


lr = LogisticRegression()
parameters = {
    'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

# Cross Validation : 5 Kfolds

cv = GridSearchCV(lr, parameters, cv=5)
cv.fit(tr_features, tr_labels.values.ravel())
print(print_results(cv))