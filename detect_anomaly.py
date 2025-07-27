import pandas as pd
from sklearn.ensemble import IsolationForest

#We load clean data (only regular transactions)
df_normal = pd.read_csv("data/clean_data.csv", sep=";")

# We train the model with cleaned data
model = IsolationForest(random_state=42)
model.fit(df_normal)

# For the example, we load the original data (with regular transaction and frauds...)
df = pd.read_csv('data/creditcard.csv')
X = df.drop(columns=['Class'])


# We isolate a single line
row_index = 541
transaction = df.iloc[row_index].drop("Class").to_frame().T

# We use the model to predict the result
result = model.predict(transaction)[0] == -1


# We display the result in the console
print(df.iloc[row_index])
print(f"Transaction #{row_index} → {'Fraud detected' if result else '✅ Normal transaction'}")