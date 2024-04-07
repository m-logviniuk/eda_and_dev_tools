import pickle
import pandas as pd
from explainerdashboard import *

if __name__ == '__main__':
    pkl_file = "svr_best.pkl"

    with open(pkl_file, 'rb') as file:
        svr_best = pickle.load(file)

    df_test = pd.read_csv('test_preprocessed.csv')

    X_test = df_test.drop(['Age'], axis=1)
    y_test = df_test['Age']

    explainer = RegressionExplainer(svr_best, X_test, y_test)

    db = ExplainerDashboard(explainer)

    db.to_yaml("dashboard.yaml", explainerfile="explainer.dill", dump_explainer=True)
