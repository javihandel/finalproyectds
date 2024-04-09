import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import requests
from io import BytesIO
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import accuracy_score, plot_confusion_matrix


URL1 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
resp = requests.get(URL1)
data = pd.read_csv(BytesIO(resp.content))
data.head(5)

URL2 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_3.csv'
resp2 = requests.get(URL2)
x = pd.read_csv(BytesIO(resp.content))
x.head(5)

# Seleccionar solo las columnas numéricas para el preprocesamiento
numeric_cols = x.select_dtypes(include=['int64', 'float64']).columns
x = x[numeric_cols]

# 1
Y = data['Class'].to_numpy()
print(type(Y)) 

# 2
# Estandarizar los datos
scaler = StandardScaler()
x = scaler.fit_transform(x)

# 3
X_train, X_test, Y_train, Y_test = train_test_split(x, Y, test_size=0.2, random_state=2)
Y_test.shape

# 4
# Crear objeto de regresión logística
lr = LogisticRegression()
# Definir los parámetros para la búsqueda de cuadrícula
parameters = {'C': [0.01, 0.1, 1], 'penalty': ['l2'], 'solver': ['lbfgs']}
# Crear objeto GridSearchCV para regresión logística
logreg_cv = GridSearchCV(lr, parameters, cv=10)
# Ajustar el objeto GridSearchCV para encontrar los mejores parámetros
logreg_cv.fit(X_train, Y_train)
# Mostrar los mejores parámetros y la precisión en los datos de validación
print("Parámetros ajustados (mejores parámetros): ", logreg_cv.best_params_)
print("Precisión: ", logreg_cv.best_score_)

Parámetros ajustados (mejores parámetros):  {'C': 1, 'penalty': 'l2', 'solver': 'lbfgs'}
Precisión:  1.0

# 5
# Calcular la precisión en los datos de prueba
accuracy = logreg_cv.score(X_test, Y_test)
print("Precisión en los datos de prueba:", accuracy)
# Calcular la matriz de confusión
yhat = logreg_cv.predict(X_test)
conf_matrix = confusion_matrix(Y_test, yhat)
print("Matriz de confusión:\n", conf_matrix)
# Visualizar la matriz de confusión
plot_confusion_matrix(logreg_cv, X_test, Y_test)
plt.title('Matriz de Confusión')
plt.show()

Precisión en los datos de prueba: 1.0
Matriz de confusión:
 [[ 6  0]
 [ 0 12]]
C:\Users\ACER\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.7_qbz5n2kfra8p0\LocalCache\local-packages\Python37\site-packages\sklearn\utils\deprecation.py:87: FutureWarning: Function plot_confusion_matrix is deprecated; Function `plot_confusion_matrix` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: ConfusionMatrixDisplay.from_predictions or ConfusionMatrixDisplay.from_estimator.
  warnings.warn(msg, category=FutureWarning)

# 6
# Crear el objeto de Máquina de Vectores de Soporte (SVM)
svm = SVC()
# Definir los parámetros para la búsqueda de cuadrícula
parameters = {'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
              'C': np.logspace(-3, 3, 5),
              'gamma': np.logspace(-3, 3, 5)}
# Crear objeto GridSearchCV para SVM
svm_cv = GridSearchCV(svm, parameters, cv=10)
# Ajustar el objeto GridSearchCV para encontrar los mejores parámetros
svm_cv.fit(X_train, Y_train)
# Mostrar los mejores parámetros y la precisión en los datos de validación
print("Parámetros ajustados (mejores parámetros) para SVM:", svm_cv.best_params_)
print("Precisión para SVM:", svm_cv.best_score_)

Parámetros ajustados (mejores parámetros) para SVM: {'C': 0.001, 'gamma': 31.622776601683793, 'kernel': 'poly'}
Precisión para SVM: 1.0

# 7
# Calcular la precisión en los datos de prueba
accuracy = accuracy_score(Y_test, svm_cv.predict(X_test))
print("Precisión en los datos de prueba:", accuracy)
# Plotear la matriz de confusión
plot_confusion_matrix(svm_cv, X_test, Y_test)
plt.title('Matriz de Confusión para SVM')
plt.show()

Precisión en los datos de prueba: 1.0
C:\Users\ACER\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.7_qbz5n2kfra8p0\LocalCache\local-packages\Python37\site-packages\sklearn\utils\deprecation.py:87: FutureWarning: Function plot_confusion_matrix is deprecated; Function `plot_confusion_matrix` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: ConfusionMatrixDisplay.from_predictions or ConfusionMatrixDisplay.from_estimator.
  warnings.warn(msg, category=FutureWarning)

# 8
# Definir los parámetros para la búsqueda de cuadrícula
parameters = {'criterion': ['gini', 'entropy'],
              'splitter': ['best', 'random'],
              'max_depth': [2 * n for n in range(1, 10)],
              'max_features': ['auto', 'sqrt'],
              'min_samples_leaf': [1, 2, 4],
              'min_samples_split': [2, 5, 10]}
# Crear objeto de árbol de decisiones
tree = DecisionTreeClassifier()
# Crear objeto GridSearchCV para árbol de decisiones
tree_cv = GridSearchCV(tree, parameters, cv=10)
# Ajustar el objeto GridSearchCV para encontrar los mejores parámetros
tree_cv.fit(X_train, Y_train)
# Mostrar los mejores parámetros y la precisión en los datos de validación
print("Parámetros ajustados (mejores parámetros) para Árbol de Decisiones:", tree_cv.best_params_)
print("Precisión para Árbol de Decisiones:", tree_cv.best_score_)

Parámetros ajustados (mejores parámetros) para Árbol de Decisiones: {'criterion': 'gini', 'max_depth': 6, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'splitter': 'random'}
Precisión para Árbol de Decisiones: 1.0

# 9
# Calcular la precisión en los datos de prueba
accuracy_tree = tree_cv.score(X_test, Y_test)
print("Precisión en los datos de prueba para Árbol de Decisiones:", accuracy_tree)
# Calcular la matriz de confusión
yhat_tree = tree_cv.predict(X_test)
conf_matrix_tree = confusion_matrix(Y_test, yhat_tree)
print("Matriz de confusión para Árbol de Decisiones:\n", conf_matrix_tree)
# Visualizar la matriz de confusión
plot_confusion_matrix(tree_cv, X_test, Y_test)
plt.title('Matriz de Confusión para Árbol de Decisiones')
plt.show()

Precisión en los datos de prueba para Árbol de Decisiones: 0.9444444444444444
Matriz de confusión para Árbol de Decisiones:
 [[ 6  0]
 [ 1 11]]
C:\Users\ACER\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.7_qbz5n2kfra8p0\LocalCache\local-packages\Python37\site-packages\sklearn\utils\deprecation.py:87: FutureWarning: Function plot_confusion_matrix is deprecated; Function `plot_confusion_matrix` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: ConfusionMatrixDisplay.from_predictions or ConfusionMatrixDisplay.from_estimator.
  warnings.warn(msg, category=FutureWarning)

# 10
# Crear objeto KNN
KNN = KNeighborsClassifier()
# Definir los parámetros para la búsqueda de cuadrícula
parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1, 2]}
# Crear objeto GridSearchCV para KNN
knn_cv = GridSearchCV(KNN, parameters, cv=10)
# Ajustar el objeto GridSearchCV para encontrar los mejores parámetros
knn_cv.fit(X_train, Y_train)
# Mostrar los mejores parámetros y la precisión en los datos de validación
print("Parámetros ajustados (mejores parámetros): ", knn_cv.best_params_)
print("Precisión: ", knn_cv.best_score_)

Parámetros ajustados (mejores parámetros):  {'algorithm': 'auto', 'n_neighbors': 1, 'p': 2}
Precisión:  0.9857142857142858

# 11
# Calcular la precisión en los datos de prueba
accuracy_knn = knn_cv.score(X_test, Y_test)
print("Precisión en los datos de prueba para KNN:", accuracy_knn)
# Calcular la matriz de confusión para KNN
yhat_knn = knn_cv.predict(X_test)
conf_matrix_knn = confusion_matrix(Y_test, yhat_knn)
print("Matriz de confusión para KNN:\n", conf_matrix_knn)
# Visualizar la matriz de confusión para KNN
plot_confusion_matrix(knn_cv, X_test, Y_test)
plt.title('Matriz de Confusión para KNN')
plt.show()

Precisión en los datos de prueba para KNN: 1.0
Matriz de confusión para KNN:
 [[ 6  0]
 [ 0 12]]
C:\Users\ACER\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.7_qbz5n2kfra8p0\LocalCache\local-packages\Python37\site-packages\sklearn\utils\deprecation.py:87: FutureWarning: Function plot_confusion_matrix is deprecated; Function `plot_confusion_matrix` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: ConfusionMatrixDisplay.from_predictions or ConfusionMatrixDisplay.from_estimator.
  warnings.warn(msg, category=FutureWarning)










