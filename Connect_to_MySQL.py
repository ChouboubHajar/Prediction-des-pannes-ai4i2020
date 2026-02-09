# Import modules
import pandas as pd
import mysql.connector

# 1/ Load the CSV dataset
df = pd.read_csv(r"C:\Users\Hajar\dataset_projet_MySQL.csv")

# 2/ Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="0000", 
    database="prediction_machine_failure_ai4i2020"
)
cursor = conn.cursor()

# 3/ Insert each row into the table
for _, row in df.iterrows():
    sql = """
    INSERT INTO machine_data
    (UDI, Product_ID, Air_temperature_K, Process_temperature_K, Rotational_speed_rpm, Torque_Nm, Tool_wear_min,
     Temp_diff_K, power_factor_W, Tool_wear_norm_min, Type_L, Type_M, Type_H, Machine_failure,
     TWF, HDF, OSF, PWF, RNF)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(sql, tuple(row))

# 4/ Commit the changes and close the connection
conn.commit()
cursor.close()
conn.close()

print("Task finished successfully!")
