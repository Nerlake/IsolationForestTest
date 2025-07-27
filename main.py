import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/creditcard.csv')
X = df.drop(columns=['Class'])

model = IsolationForest(contamination=0.0017, random_state=42)
model.fit(X)

df["prediction"] = model.predict(X)
df["fraud_detected"] = df["prediction"] == -1
df["fraud_actual"] = df["Class"] == 1

print("Actual frauds:", df["Class"].sum())
print("Frauds detected by Isolation Forest:", df["fraud_detected"].sum())

confusion = pd.crosstab(df["fraud_actual"], df["fraud_detected"], rownames=["Actual"], colnames=["Detected"])
print(confusion)

sns.scatterplot(data=df.sample(1000), x="V2", y="V3", hue="fraud_detected", style="fraud_actual")
plt.title("Fraud Detection with Isolation Forest")
plt.show()
