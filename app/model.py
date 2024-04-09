import pickle
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

df_train = pd.read_csv('train_preprocessed.csv')

X_train = df_train.drop('Age', axis=1)
y_train = df_train["Age"]

model = SVR()

params = {'C': [0.1, 1, 10, 100],
          'gamma': [0.01, 0.1, 1, 10],
          'epsilon': [0.01, 0.1, 0.5, 1]}

gs = GridSearchCV(model, params, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1, verbose=3)
gs.fit(X_train, y_train)

svr_best = gs.best_estimator_

pkl_file = "svr_best.pkl"

with open(pkl_file, 'wb') as file:
    pickle.dump(svr_best, file)
