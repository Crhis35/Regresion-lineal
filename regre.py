import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#IMPORTAR BASE DE DATOS ONLINE
hotel = datasets.load_boston()
X = hotel.data[:, np.newaxis, 5]
Y = hotel.target


#GRAFICO DE LA GRAFICA
plt.scatter(X, Y,label='Datos')
plt.title('Hotel Hilton')
plt.xlabel('Numero de habitaciones')
plt.ylabel('Costo en dolares')
plt.grid()
plt.legend()
plt.show()

#VALORES PARA ENTENAR
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25)

#SE DEFINE REGRESION LINEAL Y LOS VALORES DE APRENDIZAJE
regr = LinearRegression()
regr.fit(x_train, y_train)

#LA RECTA PREDECIDA
y_pred = regr.predict(x_test)

#GRAFICA
plt.scatter(x_train, y_train, color = 'blue', label="Datos")
plt.scatter(x_test, y_pred, color = 'red')
plt.plot(x_train, regr.predict(x_train), color = 'Purple', label='Predicion')
plt.title('Hotel Hilton')
plt.xlabel('Numero de habitaciones')
plt.ylabel('Costo en dolares')
plt.grid()
plt.legend()
plt.show()

#VALOR DE LA PENDIENTE Y DEMAS
print("El valor de la pendiente: {0} \nValor de la intercepccion: {1} \nLa ecuacion es y={2}{3} \nLa precision del modelo = {4}".format(regr.coef_,regr.intercept_,regr.coef_,regr.intercept_,regr.score(x_train, y_train)))



