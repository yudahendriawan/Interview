import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

df = pd.read_csv("train.csv")
df.head()

df.info()

df.isna().sum()

df.drop("Cabin", axis=1, inplace=True)

df['Embarked'].value_counts()

df['Age'].fillna(df['Age'].mean(), inplace=True)

df.info()

df.isna().sum()

df.dropna(axis=0, inplace=True)

df_clean = df.copy()

df_clean = df.drop(['Name', 'Ticket'], axis=1)

df_clean = pd.get_dummies(df_clean)

X_train = df_clean.drop('Survived', axis=1)
y_train = df_clean['Survived']

# check imbalanced
y_train.value_counts()

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

model.score(X_train,y_train)

f1_score(y_train, model.predict(X_train))

classification_report(y_train, model.predict(X_train))

accuracy_score(y_train, model.predict(X_train))

df_test = pd.read_csv("test.csv")
df_test.head()

df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

X_train.columns

df_test = pd.get_dummies(df_test)
df_test.columns

predictions = model.predict(df_test)

predictions