import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

abalone_data = "https://raw.githubusercontent.com/aiedu-courses/stepik_eda_and_dev_tools/main/datasets/abalone.csv"
df = pd.read_csv(abalone_data)

df.rename(columns={'Whole weight': 'Whole_weight',
                   'Shucked weight': 'Shucked_weight',
                   'Viscera weight': 'Viscera_weight',
                   'Shell weight': 'Shell_weight'}, inplace=True)

df['Sex'] = df['Sex'].replace('f', 'F')

df.fillna({"Diameter": df["Diameter"].median(),
           "Whole_weight": df["Whole_weight"].median(),
           "Shell_weight": df["Shell_weight"].median()}, inplace=True)

df['Height'] = df['Height'].replace(0.0, df.Height.median())

df["Age"] = df["Rings"] + 1.5

del df["Rings"]

X = df.drop('Age', axis=1)
y = df["Age"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical = ['Sex']
numeric_features = ['Length', 'Diameter', 'Height', 'Whole_weight',
                    'Shucked_weight', 'Viscera_weight', 'Shell_weight']

ct = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown="ignore"), categorical),
    ('scaling', MinMaxScaler(), numeric_features)
])

X_train_transformed = ct.fit_transform(X_train)
X_test_transformed = ct.transform(X_test)

new_features = list(ct.named_transformers_['ohe'].get_feature_names_out())
new_features.extend(numeric_features)

X_train_transformed = pd.DataFrame(X_train_transformed, columns=new_features)
X_test_transformed = pd.DataFrame(X_test_transformed, columns=new_features)

train_preprocessed = pd.concat([X_train_transformed, y_train.reset_index(drop=True)], axis=1)
test_preprocessed = pd.concat([X_test_transformed, y_test.reset_index(drop=True)], axis=1)

train_preprocessed.to_csv('train_preprocessed.csv', index=False)
test_preprocessed.to_csv('test_preprocessed.csv', index=False)
