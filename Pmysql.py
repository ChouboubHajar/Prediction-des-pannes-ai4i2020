# Descriptive analytics and Predictive analytics

# Import modules
import pandas as pd
from math import pi
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import mysql.connector

#1/Load dataset
df = pd.read_csv(r"c:\Users\Hajar\Downloads\ai4i+2020+predictive+maintenance+dataset\ai4i2020.csv")


#2/Prepare data
#2-1/ One-Hot Encoding for "Type"
df["Type_L"] = (df["Type"] == "L").astype(int)
df["Type_M"] = (df["Type"] == "M").astype(int)
df["Type_H"] = (df["Type"] == "H").astype(int)

#2-2/ Delete original column
df = df.drop(columns=["Type"])

#2-3 Add derived features
df["Temp_diff [K]"] = df["Process temperature [K]"] - df["Air temperature [K]"]
df["power_factor [W]"] = (2*pi*df["Rotational speed [rpm]"]/60) * df["Torque [Nm]"]
df["Tool_wear_norm [min]"] = df["Tool wear [min]"] / df["Tool wear [min]"].max()

#2-4/ Define features (X) and target (y)
X = df.drop(columns=["UDI", "Product ID", "Machine failure", "TWF", 'HDF', "OSF", "PWF", "RNF"])
y = df["Machine failure"]



#3/ Train and test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#4/ Model training
model = LogisticRegression(class_weight="balanced", max_iter=300, random_state=42)
model.fit(X_train, y_train)


#5/ Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Coefficients:", model.coef_)


#6/ Export historical dataset (ai4i2020) for Power BI
df.to_csv("dataset_projet_MySQL.csv", index=False)


#7/ Predict new data
#7-1/ Create new data
new_data = pd.DataFrame({
    "Air temperature [K]": [25, 30, 28, 27, 32],
    "Process temperature [K]": [75, 80, 78, 77, 85],
    "Rotational speed [rpm]": [1500, 1400, 1450, 1500, 1380],
    "Torque [Nm]": [10, 12, 11, 10, 13],
    "Tool wear [min]": [50, 70, 60, 55, 80],
    "Type_L": [1, 0, 0, 1, 0],
    "Type_M": [0, 1, 0, 0, 1],
    "Type_H": [0, 0, 1, 0, 0]
})

#7-2/ Derived features for new data
new_data["Temp_diff [K]"] = new_data["Process temperature [K]"] - new_data["Air temperature [K]"]
new_data["power_factor [W]"] = (2*pi*new_data["Rotational speed [rpm]"]/60) * new_data["Torque [Nm]"]
new_data["Tool_wear_norm [min]"] = new_data["Tool wear [min]"] / df["Tool wear [min]"].max()

#7-3/ Normalize new data
new_data_scaled = scaler.transform(new_data)

#7-4/ Predict
new_data["Predicted_failure"] = model.predict(new_data_scaled)
new_data["Failure_probability"] = model.predict_proba(new_data_scaled)[:,1]


#8/ Display 
print(new_data)















