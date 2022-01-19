# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 10:02:39 2019

@author: SaezaJ01

Deep Neural Network
https://www.aprendemachinelearning.com/crear-una-red-neuronal-en-python-desde-cero/
https://github.com/jbagnato/machine-learning
"""
import numpy as np
import matplotlib.pyplot as plt
import xlrd 
import pandas as pd
from sklearn.model_selection import train_test_split



def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_derivada(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_derivada(x):
    return 1.0 - x**2


class NeuralNetwork:

    def __init__(self, layers, activation='tanh'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_derivada
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_derivada

        # inicializo los pesos
        self.weights = []
        self.deltas = []
        # capas = [2,3,2]
        # rando de pesos varia entre (-1,1)
        # asigno valores aleatorios a capa de entrada y capa oculta
        for i in range(1, len(layers) - 1):
            r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
            self.weights.append(r)
        # asigno aleatorios a capa de salida
        r = 2*np.random.random( (layers[i] + 1, layers[i+1])) - 1
        self.weights.append(r)

    def fit(self, X, y, learning_rate=0.2, epochs=100000):
        # Agrego columna de unos a las entradas X
        # Con esto agregamos la unidad de Bias a la capa de entrada
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
        
        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                    dot_value = np.dot(a[l], self.weights[l])
                    activation = self.activation(dot_value)
                    a.append(activation)
            # Calculo la diferencia en la capa de salida y el valor obtenido
            error = y[i] - a[-1]
            deltas = [error * self.activation_prime(a[-1])]
            
            # Empezamos en el segundo layer hasta el ultimo
            # (Una capa anterior a la de salida)
            for l in range(len(a) - 2, 0, -1): 
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_prime(a[l]))
            self.deltas.append(deltas)

            # invertir
            # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
            deltas.reverse()

            # backpropagation
            # 1. Multiplcar los delta de salida con las activaciones de entrada 
            #    para obtener el gradiente del peso.
            # 2. actualizo el peso restandole un porcentaje del gradiente
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

            if k % 10000 == 0: print('epochs:', k)

    def predict(self, x): 
        ones = np.atleast_2d(np.ones(x.shape[0]))
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=0)
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

    def print_weights(self):
        print("LISTADO PESOS DE CONEXIONES")
        for i in range(len(self.weights)):
            print(self.weights[i])

    def get_deltas(self):
        return self.deltas
    
    
# funcion analisis de datos y respuesta diagnostico
nn = NeuralNetwork([26,50,1],activation ='tanh') #creamos una red a nuestra medida, con 5 neuronas de entrada, 8 ocultas y 1 de salida


#Datos
parameters = [] # to store parameters names
diagnosis = []

# Give the location of the file 
loc = (r"C:\Users\jesus\Documents\python projects\mike\recoleccioìn de datos 2018.xls") 
  
# To open Workbook 
wb = xlrd.open_workbook(loc) 
sheet = wb.sheet_by_index(0)

ncols = sheet.ncols
nrows = sheet.nrows
data = [[] for i in range(1, ncols)]
patients = [[] for i in range(1, nrows)]

# For row 0 and column 0 
sheet.cell_value(0, 0) 


for i in range(sheet.ncols-1): #ultima columna es el diagnostico
    parameters.append(sheet.cell_value(0, i)) 

# asi cojo los datos por parametro, 
for i in range(1,sheet.nrows): #primera fila es de encabezados
    for j in range(1,sheet.ncols+1):
        if j==ncols:
            diagnosis.append([sheet.cell_value(i, j-1)])
        else:
            data[j-1].append(sheet.cell_value(i, j-1))
            
# cojo los datos por paciente   
for i in range(len(data[1])):
    for j in range(len(data)):

        patients[i].append(float(data[j][i]))

X = np.array(patients)       
        
#Salidas
y = np.array(diagnosis)

#dividir los datos para entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 
    
#Realizar el entrenamiento
nn.fit(X_train, y_train, learning_rate=0.003,epochs=10001)



#Z =np.array([[ 1.    ,      0.93738791 , 0.36859421 , 0.92342702 , 0.        ],
#            [ 0.89027099 , 0.536851  ,  0.61751809 , 0.47533266 , 0.2193274 ],
#            [ 0.51594215 , 0.24080466 , 0.37821391 , 0.23078362 , 0.99774974]]) 
#
#index=0
#for e in Z:
#    print("Z:",e,"Network:",nn.predict(e))
#    index=index+1

index=0
print('\n Training results:')
for e in X_train:
    print("y:",y_train[index],"Network:",nn.predict(e))
    index=index+1

# Print resultado del entrenamiento  
deltas = nn.get_deltas()
valores=[]
for arreglo in deltas:
    valores.append(arreglo[1][0])
    
index=0
print('\n Validation results:')
for e in X_test:
    print("y:",y_test[index],"Network:",nn.predict(e))
    index=index+1    
 
plt.plot(range(len(valores)), valores, color='b')
plt.ylim([0, 1])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.tight_layout()
plt.show()