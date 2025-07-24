import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('creditcard.csv')

X = dataset.drop(columns=['Class'])
model = IsolationForest(contamination=0.0017, random_state=42)
model.fit(X)

dataset["pred"] = model.predict(X)

dataset["fraude_detectee"] = dataset["pred"] == -1
dataset["fraude_reelle"] = dataset["Class"] == 1


print("Fraudes réelles :", dataset['Class'].sum())
print("Fraudes détectées par Isolation Forest :", dataset["fraude_detectee"].sum())

confusion = pd.crosstab(dataset["fraude_reelle"], dataset["fraude_detectee"], rownames=["Réel"], colnames=["Détecté"])
print(confusion)

sns.scatterplot(data=dataset.sample(1000), x="V2", y="V3", hue="fraude_detectee", style="fraude_reelle")
plt.title("Détection de fraudes avec Isolation Forest")
plt.show()

