# pour manipuler le tableau (dataset)
import pandas as pd
from math import pi
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

#Lire le dataset .csv
df=pd.read_csv(r"c:\Users\Hajar\Downloads\ai4i+2020+predictive+maintenance+dataset\ai4i2020.csv")

#Types
#L (6000)    M (2997)    H (1003)



#Changer les types de str ---->int pour le modele ia (il travaille que sur les int only)
#Explication
#[23:35, 29/01/2026] Hajar: Alors pourquoi on parle de 0 / 1 ?
#UNIQUEMENT pour les modèles de Machine Learning
#Parce que :
#un algorithme ne comprend pas le texte
# comprend seulement les nombres par contre POWER BI contient des nombres ,texte et categories
#[23:36, 29/01/2026] Hajar: 4. Mais attention (très important)
# Mauvais pour le ML
#L = 0, M = 1, H = 2 (le dataset de mon projet)
#Pourquoi ?
# le modèle croit que H > M > L
# souvent faux
# Bon pour le ML
#Transformer en plusieurs colonnes 0/1 :
#Type_L  Type_M  Type_H
#1       0       0
#0       1       0
#0       0       1
#Ça s’appelle one-hot encoding.
#5. Résumé en 3 lignes (à mémoriser)
#Power BI → garder le texte
# → garder le texte
#Machine Learning → transformer en 0/1
#6. Exemple ultra concret
#Même donnée, deux usages :
#Pour Power BI
#Type = "Low"
#Panne = "Panne"
#Pour ML
#Type_L = 1
#Type_M = 0
#Type_H = 0
#Panne = 1
df["Type_L"]=[1 if x=="L" else 0 for x in df["Type"]]
df["Type_M"]=[1 if x=="M" else 0 for x in df["Type"]]
df["Type_H"]=[1 if x=="H" else 0 for x in df["Type"]]

#Supprimer la colonne Type
df = df.drop(columns=["Type"])

#Ajout des colonnes utiles
df["Temp_diff [K]"]=df["Process temperature [K]"]-df["Air temperature [K]"] 
df["power_factor [W]"]=(2*pi*df["Rotational speed [rpm]"]/60)*df["Torque [Nm]"] 
df['Tool_wear_norm [min]'] = df['Tool wear [min]'] / df['Tool wear [min]'].max()


#Partie du modele IA
#le panne (Machine failure)
#0 9661 (Pas de panne)
#1 339  (Panne)

#Types
#L (6000)    M (2997)    H (1003)

#Changer les types de str ---->int pour le modele ia (il travaille que sur les int only)
#Explication
#[23:35, 29/01/2026] Hajar: Alors pourquoi on parle de 0 / 1 ?
#UNIQUEMENT pour les modèles de Machine Learning
#Parce que :
#un algorithme ne comprend pas le texte
# comprend seulement les nombres par contre POWER BI contient des nombres ,texte et categories
#[23:36, 29/01/2026] Hajar: 4. Mais attention (très important)
# Mauvais pour le ML
#L = 0, M = 1, H = 2 (le dataset de mon projet)
#Pourquoi ?
# le modèle croit que H > M > L
# souvent faux
# Bon pour le ML
#Transformer en plusieurs colonnes 0/1 :
#Type_L  Type_M  Type_H
#1       0       0
#0       1       0
#0       0       1
#Ça s’appelle one-hot encoding.
#5. Résumé en 3 lignes (à mémoriser)
#Power BI → garder le texte
# → garder le texte
#Machine Learning → transformer en 0/1
#6. Exemple ultra concret
#Même donnée, deux usages :
#Pour Power BI
#Type = "Low"
#Panne = "Panne"
#Pour ML
#Type_L = 1
#Type_M = 0
#Type_H = 0
#Panne = 1  

#X les entrees y est la sortie 0/1 (panne/pas de panne)
X=df.drop(columns=["UDI","Product ID","Machine failure","TWF",'HDF',"OSF","PWF","RNF"])
y=df["Machine failure"]

#Partie de ia
# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# normalisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# modèle amélioré
model = LogisticRegression(
    class_weight="balanced",
    max_iter=300,
    random_state=42
)

# entraînement
model.fit(X_train, y_train)

# prédiction
y_pred = model.predict(X_test)

# évaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# coefficients
print("Coefficients du modèle:", model.coef_)

#Exporter le dataset 
df.to_csv("dataset_projet_MySQL.csv",index=False)



#Predire les pannes sur des nouvelles donnees
#Exemple de nouvelles observations ---
new_data = pd.DataFrame({
    "Air temperature [K]": [25, 30, 28, 27, 32],
    "Process temperature [K]": [1375, 1380, 1578, 1377, 1285],
    "Rotational speed [rpm]": [150, 100, 145, 150, 380],
    "Torque [Nm]": [10, 12, 11, 10, 13],
    "Tool wear [min]": [234, 370, 190, 355, 280],
    "Type_L": [1, 0, 0, 1, 0],
    "Type_M": [0, 1, 0, 0, 1],
    "Type_H": [0, 0, 1, 0, 0]   
})

# Calcul des colonnes supplémentaires comme dans le dataset 
new_data["Temp_diff [K]"] = new_data["Process temperature [K]"] - new_data["Air temperature [K]"]
new_data["power_factor [W]"] = (2*pi*new_data["Rotational speed [rpm]"]/60) * new_data["Torque [Nm]"]
new_data["Tool_wear_norm [min]"] = new_data["Tool wear [min]"] / df['Tool wear [min]'].max()  # même max que dataset historique
# Normalisation avec scaler déjà entraîné
new_data_scaled = scaler.transform(new_data)  # NE PAS faire fit_transform
# Prédiction avec ton modèle 
predictions = model.predict(new_data_scaled)
probabilities = model.predict_proba(new_data_scaled)[:,1]  # probabilité de panne
# Ajouter les résultats au DataFrame 
new_data["Predicted_failure"] = predictions
new_data["Failure_probability"] = probabilities
# Export pour Power BI 
new_data.to_csv("New_Observations_Predictions.csv", index=False)
# Affichage rapide pour vérification
print(new_data)














