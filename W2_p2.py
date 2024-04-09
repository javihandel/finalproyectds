import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import BytesIO
from sklearn.preprocessing import OneHotEncoder


# Código
URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
resp = requests.get(URL)
df = pd.read_csv(BytesIO(resp.content))
df.head(5)

sns.catplot(y="PayloadMass", x="FlightNumber", hue="Class", data=df, aspect = 5)
plt.xlabel("Flight Number",fontsize=20)
plt.ylabel("Pay load Mass (kg)",fontsize=20)
plt.show()

# 1
sns.catplot(x="FlightNumber", y="LaunchSite", hue="Class", data=df, kind="bar", height=6, aspect=2)
plt.xlabel("Número de vuelo", fontsize=14)
plt.ylabel("Sitio de lanzamiento", fontsize=14)
plt.title("Relación entre número de vuelo y sitio de lanzamiento", fontsize=16)
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x="FlightNumber", y="LaunchSite", hue="Class", data=df)
plt.xlabel("Número de vuelo", fontsize=14)
plt.ylabel("Sitio de lanzamiento", fontsize=14)
plt.title("Gráfico de dispersión: Número de vuelo vs. Sitio de lanzamiento (con Clase)", fontsize=16)
plt.legend(title="Clase", loc="upper right")
plt.show()

# 2
plt.figure(figsize=(10, 6))
sns.scatterplot(x="PayloadMass", y="LaunchSite", data=df)
plt.xlabel("Carga útil (Payload)", fontsize=14)
plt.ylabel("Sitio de lanzamiento", fontsize=14)
plt.title("Relación entre carga útil y sitio de lanzamiento", fontsize=16)
plt.show()
plt.figure(figsize=(10, 6))
sns.barplot(x="PayloadMass", y="LaunchSite", data=df)
plt.xlabel("Carga útil (Payload)", fontsize=14)
plt.ylabel("Sitio de lanzamiento", fontsize=14)
plt.title("Relación entre carga útil y sitio de lanzamiento", fontsize=16)
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x="PayloadMass", y="LaunchSite", hue="Class", data=df)
plt.xlabel("Masa de carga útil (kg)", fontsize=14)
plt.ylabel("Sitio de lanzamiento", fontsize=14)
plt.title("Gráfico de dispersión: Masa de carga útil vs. Sitio de lanzamiento (con Clase)", fontsize=16)
plt.legend(title="Clase", loc="upper right")
plt.show()

# 3
# Calcula la tasa de éxito para cada tipo de órbita
success_rate = df.groupby('Orbit')['Class'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='Orbit', y='Class', data=success_rate, palette='viridis')
plt.xlabel('Tipo de órbita', fontsize=14)
plt.ylabel('Tasa de éxito', fontsize=14)
plt.title('Relación entre tasa de éxito y tipo de órbita', fontsize=16)
plt.xticks(rotation=45)
plt.show()

# 4

plt.figure(figsize=(10, 6))
sns.scatterplot(x='FlightNumber', y='Orbit', data=df)
plt.xlabel('Número de vuelo', fontsize=14)
plt.ylabel('Tipo de órbita', fontsize=14)
plt.title('Relación entre número de vuelo y tipo de órbita', fontsize=16)
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='FlightNumber', y='Orbit', hue='Class', data=df)
plt.xlabel('Número de vuelo', fontsize=14)
plt.ylabel('Órbita', fontsize=14)
plt.title('Gráfico de dispersión: Número de vuelo vs. Órbita (con Clase)', fontsize=16)
plt.legend(title='Clase', loc='upper right')
plt.show()

# 5

plt.figure(figsize=(10, 6))
sns.boxplot(x='Orbit', y='PayloadMass', data=df)
plt.xlabel('Tipo de órbita', fontsize=14)
plt.ylabel('Carga útil (Payload)', fontsize=14)
plt.title('Relación entre carga útil y tipo de órbita', fontsize=16)
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='PayloadMass', y='Orbit', hue='Class', data=df)
plt.xlabel('Carga útil (Payload)', fontsize=14)
plt.ylabel('Órbita', fontsize=14)
plt.title('Gráfico de dispersión: Carga útil vs. Órbita (con Clase)', fontsize=16)
plt.legend(title='Clase', loc='upper right')
plt.show()

# 6

year = []

def Extract_year():
    global year
    for date in df["Date"]:
        year.append(date.split("-")[0])  # Extraer el año de la fecha y agregarlo a la lista
    return year
Extract_year()
df['Year'] = year
df.head()

success_rate_yearly = df.groupby('Year')['Class'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(x='Year', y='Class', data=success_rate_yearly)
plt.xlabel('Año', fontsize=14)
plt.ylabel('Tasa de éxito promedio', fontsize=14)
plt.title('Tendencia anual de éxito de lanzamientos', fontsize=16)
plt.show()

# Código
features = df[['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights', 'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial']]
features.head()

# 7
# Seleccionar las columnas categóricas para crear variables ficticias
categorical_cols = ['Orbit', 'LaunchSite', 'LandingPad', 'Serial']
# Crear variables ficticias utilizando get_dummies
dummy_cols = pd.get_dummies(features[categorical_cols], drop_first=True)
# Crear el objeto OneHotEncoder
encoder = OneHotEncoder(sparse=False, drop='first')
# Ajustar y transformar las columnas categóricas seleccionadas con OneHotEncoder
encoded_cols = encoder.fit_transform(features[categorical_cols])
# Convertir el resultado en un DataFrame y asignarlo a features_one_hot
features_one_hot = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names(categorical_cols))
# Concatenar las variables ficticias con el DataFrame original
features_encoded = pd.concat([features, features_one_hot], axis=1)
# Eliminar las columnas categóricas originales
features_encoded.drop(categorical_cols, axis=1, inplace=True)
# Mostrar las primeras filas del DataFrame con las variables ficticias codificadas
features_encoded.head()


# 8
# Convertir todas las columnas numéricas a tipo float64
features_encoded = features_encoded.astype({'FlightNumber': 'float64',
                                            'PayloadMass': 'float64',
                                            'Flights': 'float64',
                                            'GridFins': 'float64',
                                            'Reused': 'float64',
                                            'Legs': 'float64',
                                            'Block': 'float64',
                                            'ReusedCount': 'float64'})

# Mostrar las primeras filas del DataFrame con las columnas convertidas
features_encoded.head()

features_encoded.to_csv('dataset_part_3.csv', index=False)


