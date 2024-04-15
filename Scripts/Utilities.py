import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#from neuralsens import partial_derivatives as ns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,mean_squared_error,mean_absolute_error, r2_score
import random
import time

# Machine learning libraries

import torch
from torch.autograd.functional import jacobian
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Softmax
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss,MSELoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.autograd.functional import hessian, jacobian

def print_errors(model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor,log=False,config=None):
    device = next(model.parameters()).device  # Obtiene el dispositivo del modelo

    # Mueve los datos a la misma GPU o CPU que el modelo
    X_train_tensor = X_train_tensor.to(device)
    X_test_tensor = X_test_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)
    y_test_tensor = y_test_tensor.to(device)

    #y_train_pred = model(X_train_tensor).detach().cpu().numpy() if device.type == 'cuda' else model(X_train_tensor).detach().numpy()
    #y_test_pred = model(X_test_tensor).detach().cpu().numpy() if device.type == 'cuda' else model(X_test_tensor).detach().numpy()
    y_train_pred = model(X_train_tensor)
    y_test_pred = model(X_test_tensor)
    mse_train = torch.mean((y_train_tensor - y_train_pred)**2)
    mse_test = torch.mean((y_test_tensor - y_test_pred)**2)
    rmse_train = torch.sqrt(mse_train)
    rmse_test = torch.sqrt(mse_test)
    mae_train = torch.mean(torch.abs(y_train_tensor - y_train_pred))
    mae_test = torch.mean(torch.abs(y_test_tensor - y_test_pred))
    r2_train = r2_score(y_train_tensor.detach().cpu().numpy(), y_train_pred.detach().cpu().numpy())
    r2_test = r2_score(y_test_tensor.detach().cpu().numpy(), y_test_pred.detach().cpu().numpy())
    

    print(f"MSE Train: {np.round(mse_train.item(),7)}, MSE Test: {np.round(mse_test.item(),7)}")
    print(f"RMSE Train: {np.round(rmse_train.item(),7)}, RMSE Test: {np.round(rmse_test.item(),7)}")
    print(f"MAE Train: {np.round(mae_train.item(),7)}, MAE Test: {np.round(mae_test.item(),7)}")
    print(f"R2 Train: {np.round(r2_train,7)}, R2 Test: {np.round(r2_test,7)}")
    

def encontrar_indices(lista, valor):
    """
    Esta función devuelve una lista de los índices en los que se encuentra el valor dado en la lista dada.
    """
    indices = []
    for i in range(len(lista)):
        if lista[i] == valor:
            indices.append(i)
    return indices


## Escalamos las columnas
def add_scaled_columns(df, columns):
    df_copy=df.copy()
    scaler = StandardScaler()
    for column in columns:
        column_name = column + '_scaled'
        scaled_column = scaler.fit_transform(df_copy[column].values.reshape(-1, 1))
        df_copy.loc[:, column_name] = scaled_column
    return df_copy 

def get_accuracy(y_true, y_prob):
    accuracy = accuracy_score(y_true, y_prob > 0.5)
    return accuracy


def generate_synthetic_tensor(original_tensor, n, mean=0, std=1):

    # Creamos un tensor vacío de tamaño [n, 20]
    synthetic_tensor = torch.empty((n, original_tensor.shape[1]))

    # Generamos un tensor con índices de fila para el tensor original
    row_indices = torch.randint(low=0, high=original_tensor.shape[0], size=(n,))

    # Copiamos las filas del tensor original
    synthetic_tensor[:, :] = original_tensor[row_indices, :]

    # Generamos un tensor de ruido con valores aleatorios de una distribución normal
    noise_tensor = torch.normal(mean=mean, std=std, size=(n, original_tensor.shape[1]))

    # Sumamos el tensor de ruido al tensor sintético
    synthetic_tensor += noise_tensor

    return synthetic_tensor

def generate_synthetic_tensor_categorical(original_tensor, n, categorical_columns, mean=0.0, std=1.0):

    # Creamos un tensor vacío de tamaño [n, 20]
    synthetic_tensor = torch.empty((n, original_tensor.shape[1]))

    # Generamos un tensor con índices de fila para el tensor original
    row_indices = torch.randint(low=0, high=original_tensor.shape[0], size=(n,))

    # Copiamos las filas del tensor original
    synthetic_tensor[:, :] = original_tensor[row_indices, :]

    # Iteramos sobre las columnas
    for columna in range(original_tensor.shape[1]):
        if columna in categorical_columns:
            # Tomar un valor aleatorio del conjunto de valores categóricos de esa columna
            #unique_values = torch.unique(original_tensor[:, columna])
            val_list = random.choices(original_tensor[:, columna],k=n)
            synthetic_tensor[:, columna] = torch.tensor(val_list)
        else:
            # Generamos un tensor de ruido con valores aleatorios de una distribución normal
            noise_tensor = torch.normal(mean=mean, std=std, size=(n,))
            synthetic_tensor[:, columna] += noise_tensor

    return synthetic_tensor

def add_timestamp(string):
    timestamp = str(int(time.time()))
    return f"{string}_{timestamp}"

def batch_hessian(model, input):
    """Calcula el jacobiano para todas las muetras de un batch. El resultado es 
    un tensor de tamaño (n_outputs,batch_size,n_inputs) de manera que los 
    siguientes vectores
    ...
    """
    f_sum = lambda x: torch.sum(model(x), axis=0)
    return hessian(f_sum, input, create_graph=True) 


import shutil
import os

def borrar_models(ruta='./Models/'):
    # Ruta de la carpeta que deseas eliminar
    #ruta = './Models/'

    # Eliminar la carpeta y su contenido
    shutil.rmtree(ruta)

    # Puedes recrear la carpeta si es necesario
    os.mkdir(ruta)