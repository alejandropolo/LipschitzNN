import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#from neuralsens import partial_derivatives as ns
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Machine learning libraries
from tqdm import tqdm
import torch
from torch.autograd.functional import jacobian
from torch.optim import SGD, Adam,LBFGS
from torch.nn import BCELoss,MSELoss
import torch.nn as nn
import plotly.graph_objs as go
import plotly.express as px
from sklearn.metrics import accuracy_score

import Utilities as Utilities
import importlib
import sys
sys.path.append('../Scripts/')
from pytorchtools import EarlyStopping
import logging
from datetime import datetime
import torch

importlib.reload(Utilities)

## GENERAR LOS LOGS
## If logs directory does not exist, create it
if not os.path.exists('../logs'):
    os.makedirs('../logs')

### Se usa una configuración básica
logging.basicConfig(filename=datetime.now().strftime('../logs/train_log_%H_%M_%d_%m_%Y.log'),
                    format='%(asctime)s %(message)s',
                    filemode='w',force=True)
 
### Creación de un objeto
logger = logging.getLogger()

### Seteado del nivel
logger.setLevel(logging.INFO)

class MLP_Monotonic:

    def __init__(self,_model_name,_model):
        self._model_name = _model_name
        self._model = _model
        self._jacobian = None
        self._avg_train_losses = None
        self._avg_train_losses_modified = None
        self._avg_valid_losses = None

    def batch_jacobian(self, input):
        """Calcula el jacobiano para todas las muetras de un batch. El resultado es 
        un tensor de tamaño (n_outputs,batch_size,n_inputs) de manera que los 
        siguientes vectores

        - jacobian[i,j,k]: Devuelve el jacobiano con respecto a la salida i y la variable k, evaluado en la muestra j
        (f/x_i)'(batch_j)

        Args:
            f: Funcion (o modelo)
            x: input

        Returns:
            Jacobian: Matriz jacobiana
        """
        f_sum = lambda x: torch.sum(self._model(x), axis=0)
        return jacobian(f_sum, input,create_graph=True) 
    
    def checkpoint(model, filename):
        torch.save(model.state_dict(), filename)
        
    def resume(model, filename):
        model.load_state_dict(torch.load(filename))

    def train_delta_decay(self,train_data,val_data,criterion,n_epochs,verbose=1,n_visualized=1,
                 monotone_relations=[0],optimizer_type='Adam',learning_rate=0.01,delta=0.5,weight_decay=0.0,
                 patience=100,model_path='./Models/checkpoint_'):
        """
        Modificación para añadir un decay en la penalizacion        
        """
        # to track the training loss as the model trains
        train_losses = []
        # to track the training loss as the model trains
        train_losses_modified = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses_modified = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = [] 

        
        # initialize the early_stopping object
        ## Generate model path
        model_path_timestamp = Utilities.add_timestamp(model_path)
        path = model_path_timestamp+'.pt'
        early_stopping = EarlyStopping(path=path,patience=patience, verbose=False)

        ## Tipo de optimizador
        if optimizer_type == 'SGD':
            optimizer = SGD(self._model.parameters(),lr=learning_rate, momentum=0.9)
        elif optimizer_type == 'Adam':
            optimizer = Adam(self._model.parameters(),lr=learning_rate,weight_decay=weight_decay)
        elif optimizer_type == 'LBFGS':
            optimizer = LBFGS(self._model.parameters(),lr=learning_rate)

        
        ## El número de variables de entrada
        n_vars = len(monotone_relations)

        ## Indices de las variables monótonas (crec o decrec)
        var_mono_crec = Utilities.encontrar_indices(monotone_relations,1)
        var_mono_decrec = Utilities.encontrar_indices(monotone_relations,-1)

        for epoch in range(n_epochs+1):
            ###################
            # train the model #
            ###################

            ## Activamos el modulo de entrenamiento
            self._model.train()

            for i, (inputs,targets) in enumerate(train_data):
                
            
                # Se resetea el gradiente
                optimizer.zero_grad()

                # Forward
                yhat = self._model(inputs)

                # Loss
                loss = criterion(yhat,targets)

                ## Se añade el coeficiente corrector
                jacob=self.batch_jacobian(inputs)

                ## Si la relacion es creciente (-relu) y si decreciente (relu)
                adjusted_loss_crec = torch.sum(torch.relu(-jacob[0, :, var_mono_crec]))
                adjusted_loss_decrec = torch.sum(torch.relu(jacob[0, :, var_mono_decrec]))

                # Ajustar el valor de delta
                delta_mod = delta / (1 + 0.05 * epoch)
                loss_modified = loss + delta_mod*(adjusted_loss_decrec+adjusted_loss_crec)
                # Backward
                loss_modified.backward()

                # Step
                if optimizer_type == 'LBFGS':
                    optimizer.step(closure=lambda: self.closure(inputs, targets, criterion, optimizer))
                else:
                    optimizer.step()
                
                # Record training loss
                train_losses.append(loss.item())
                train_losses_modified.append(loss_modified.item())

            ######################    
            # validate the model #
            ######################

            # Almacenamos la informacion de la validacion
            self._model.eval() #Switch a modo evaluacion
            
            with torch.no_grad():
                for x,y in val_data:
                    # Forward
                    out = self._model(x)
                    # Loss
                    loss_val = criterion(out,y)
                    # Record validation loss
                    valid_losses.append(loss_val.item())

            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            train_loss_modified = np.average(train_losses_modified)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_train_losses_modified.append(train_loss_modified)
            avg_valid_losses.append(valid_loss)

                    
                
            with torch.no_grad():
                ## El número de variables de entrada
                n_vars = inputs.shape[1]
                epoch_len = len(str(n_epochs))

                ## Dependiendo del tipo se printea más o menos contenido
                if verbose==2:
                    if epoch % n_visualized ==0:
                        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}]')
                        print(print_msg,end=' ')
                        print('Train Loss %f, Train Loss Mod %f, Val Loss %f' %(float(train_loss),float(train_loss_modified),float(valid_loss)),end=' ')
                        for i in range(n_vars):
                            if i in var_mono_crec:
                                print('Minimum Jacobian x_{}  %f'.format(i+1) %(float(jacob[0,:, i].min())),end=', ')
                            elif i in var_mono_decrec:
                                print('Maximum Jacobian x_{}  %f'.format(i+1) %(float(jacob[0,:, i].max())),end=', ')
                            else:
                                print('(Min Jacobian x_{}  %f, Max Jacobian x_{} %f )'.format(i+1,i+1) %(float(jacob[0,:, i].min()),float(jacob[0,:, i].max())))
                elif verbose==1:
                    print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}]')
                    print(print_msg,end=' ')
                    print('Train Loss %f, Train Loss Mod %f, Val Loss %f' %(float(train_loss),float(train_loss_modified),float(valid_loss)))
                else:
                    if epoch == n_epochs:
                        ## Only print in last epoch
                        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}]')
                        print(print_msg,end=' ')
                        print('Train Loss %f, Train Loss Mod %f, Val Loss %f' %(float(train_loss),float(train_loss_modified),float(valid_loss)))

            # clear lists to track next epoch
            train_losses = []
            train_losses_modified = []
            valid_losses = []

            
            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, self._model)

            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        # load the last checkpoint with the best model
        self._model.load_state_dict(torch.load(path))

        ## Almacenamos las curvas de entrenamiento
        self._avg_train_losses = avg_train_losses
        self._avg_train_losses_modified = avg_train_losses_modified
        self._avg_valid_losses = avg_valid_losses

        ## Almacenado en la propia clase
        self._jacobian = jacob

    def train_adjusted_std(self,train_data,val_data,criterion,n_epochs,categorical_columns=[None],verbose=1,n_visualized=1,
                 monotone_relations=[0],optimizer_type='Adam',learning_rate=0.01,delta=0.0,weight_decay=0.0,delta_synthetic=0.0,delta_external=0.0,
                 patience=100,model_path='./Models/checkpoint_',std_syntethic=0.0,std_growth=0.0,epsilon =0.0,
                 epsilon_synthetic=0.0,external_points = None,seed=None,keep_model=False):
        """
        Modificacion de la función train para que vaya aumentando la varianza
        """
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
        # to track the training loss as the model trains
        train_losses = []
        # to track the training loss as the model trains
        train_losses_modified = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses_modified = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = [] 

        
        # initialize the early_stopping object
        model_path_timestamp = Utilities.add_timestamp(model_path)
        path = model_path_timestamp+'.pt'
        early_stopping = EarlyStopping(path=path,patience=patience, verbose=False)

        ## Tipo de optimizador
        if optimizer_type == 'SGD':
            optimizer = SGD(self._model.parameters(),lr=learning_rate, momentum=0.9)
        elif optimizer_type == 'Adam':
            optimizer = Adam(self._model.parameters(),lr=learning_rate,weight_decay=weight_decay)
        elif optimizer_type == 'LBFGS':
            optimizer = LBFGS(self._model.parameters(),lr=learning_rate)

        
        ## El número de variables de entrada
        n_vars = len(monotone_relations)

        ## Indices de las variables monótonas (crec o decrec)
        var_mono_crec = Utilities.encontrar_indices(monotone_relations,1)
        var_mono_decrec = Utilities.encontrar_indices(monotone_relations,-1)

        #for epoch in range(n_epochs+1):
        pbar = tqdm(range(n_epochs+1))
        for epoch in pbar:

            ###################
            # train the model #
            ###################
            ## Activamos el modulo de entrenamiento
            self._model.train()

            #for i, (inputs,targets) in enumerate(train_data):
            for i, batch in enumerate(train_data):
                if isinstance(batch, list) and len(batch) == 3:
                    batch_input1, batch_input2, targets = batch
                    inputs = (batch_input1,batch_input2)
                else:
                    inputs, targets = batch
                if optimizer_type == 'Adam':
                    # Se resetea el gradiente
                    optimizer.zero_grad()

                    # Forward
                    yhat = self._model(inputs)

                    # Loss
                    loss = criterion(yhat,targets)

                    ## Evaluamos el jacobiano en los datos de entrenamiento

                    if delta>0:
                        ## Se añade el coeficiente corrector
                        jacob=self.batch_jacobian(inputs)
                        ## Si la relacion es creciente (-relu) y si decreciente (relu)
                        adjusted_loss_crec = torch.sum(torch.relu(-jacob[0, :, var_mono_crec]+epsilon))
                        adjusted_loss_decrec = torch.sum(torch.relu(jacob[0, :, var_mono_decrec]+epsilon))
                        
                    else:
                        adjusted_loss_crec = 0.0
                        adjusted_loss_decrec = 0.0

                    ## Evaluamos el jacobiano en datos sintéticos generados
                    if delta_synthetic>0:

                        with torch.no_grad():  
                            synthetic_data=Utilities.generate_synthetic_tensor_categorical(inputs,100,categorical_columns,mean=0,std=std_syntethic)

                        jacob_synthetic = self.batch_jacobian(synthetic_data)

                        ## Si la relacion es creciente (-relu) y si decreciente (relu)
                        adjusted_loss_crec_synthetic = torch.sum(torch.relu(-jacob_synthetic[0, :, var_mono_crec]+epsilon))
                        adjusted_loss_decrec_synthetic = torch.sum(torch.relu(jacob_synthetic[0, :, var_mono_decrec]+epsilon))
                        loss_modified_synthetic = delta_synthetic*(adjusted_loss_decrec_synthetic+adjusted_loss_crec_synthetic)
                    elif delta_synthetic==0:
                        loss_modified_synthetic = 0.0
                    
                    ##Evaluamos el jacobiano en puntos externos
                    if delta_external>0:
                        # Calcular el jacobiano en puntos externos y agregar la penalización al loss
                        jacob_external = self.batch_jacobian(external_points)
                        adjusted_loss_crec_external = torch.sum(torch.relu(-jacob_external[0, :, var_mono_crec]+epsilon))
                        adjusted_loss_decrec_external = torch.sum(torch.relu(jacob_external[0, :, var_mono_decrec]+epsilon))
                        loss_modified_external = delta_external * (adjusted_loss_decrec_external + adjusted_loss_crec_external)
                    else:
                        loss_modified_external = 0.0

                        
                    with torch.no_grad():
                        if loss_modified_synthetic < epsilon_synthetic:
                            std_syntethic = std_syntethic+std_growth
                    loss_modified = loss + delta * (adjusted_loss_decrec + adjusted_loss_crec) + loss_modified_synthetic + loss_modified_external

                def closure():
                    optimizer.zero_grad()
                    outputs = self._model(inputs)
                    loss = criterion(outputs, targets)

                    if delta>0:
                        ## Se añade el coeficiente corrector
                        jacob=self.batch_jacobian(inputs)
                        ## Si la relacion es creciente (-relu) y si decreciente (relu)
                        adjusted_loss_crec = torch.sum(torch.relu(-jacob[0, :, var_mono_crec]+epsilon))
                        adjusted_loss_decrec = torch.sum(torch.relu(jacob[0, :, var_mono_decrec]+epsilon))
                        
                    else:
                        adjusted_loss_crec = 0.0
                        adjusted_loss_decrec = 0.0

                    ## Evaluamos el jacobiano en datos sintéticos generados
                    if delta_synthetic>0:

                        with torch.no_grad():  
                            synthetic_data=Utilities.generate_synthetic_tensor_categorical(inputs,100,categorical_columns,mean=0,std=std_syntethic)

                        jacob_synthetic = self.batch_jacobian(synthetic_data)

                        ## Si la relacion es creciente (-relu) y si decreciente (relu)
                        adjusted_loss_crec_synthetic = torch.sum(torch.relu(-jacob_synthetic[0, :, var_mono_crec]+epsilon))
                        adjusted_loss_decrec_synthetic = torch.sum(torch.relu(jacob_synthetic[0, :, var_mono_decrec]+epsilon))
                        loss_modified_synthetic = delta_synthetic*(adjusted_loss_decrec_synthetic+adjusted_loss_crec_synthetic)
                    elif delta_synthetic==0:
                        loss_modified_synthetic = 0.0
                    
                    ##Evaluamos el jacobiano en puntos externos
                    if delta_external>0:
                        # Calcular el jacobiano en puntos externos y agregar la penalización al loss
                        jacob_external = self.batch_jacobian(external_points)
                        adjusted_loss_crec_external = torch.sum(torch.relu(-jacob_external[0, :, var_mono_crec]+epsilon))
                        adjusted_loss_decrec_external = torch.sum(torch.relu(jacob_external[0, :, var_mono_decrec]+epsilon))
                        loss_modified_external = delta_external * (adjusted_loss_decrec_external + adjusted_loss_crec_external)
                    else:
                        loss_modified_external = 0.0
                    with torch.no_grad():
                        if loss_modified_synthetic < epsilon_synthetic:
                            std_syntethic = std_syntethic+std_growth
                    loss_modified = loss + delta * (adjusted_loss_decrec + adjusted_loss_crec) + loss_modified_synthetic + loss_modified_external
                    loss_modified.backward()
                    # Record training loss
                    train_losses.append(loss.item())
                    train_losses_modified.append(loss_modified.item())
                    return loss_modified
                
                # Step
                if optimizer_type=='LBFGS':
                    #optimizer.step(closure=lambda: self.closure(inputs, targets, criterion, optimizer))
                    optimizer.step(closure)
                else:
                    loss_modified.backward()
                    optimizer.step()
                    # Record training loss
                    train_losses.append(loss.item())
                    train_losses_modified.append(loss_modified.item())
                

            ######################    
            # validate the model #
            ######################

            # Almacenamos la informacion de la validacion
            self._model.eval() #Switch a modo evaluacion
            with torch.no_grad():
                #for x,y in val_data:
                for batch in val_data:
                    if isinstance(batch, list) and len(batch) == 3:
                        x1, x2, y = batch
                        x = (x1,x2)
                    else:
                        x, y = batch
                    # Forward
                    out = self._model(x)
                    # Loss
                    loss_val = criterion(out,y)
                    # Record validation loss
                    valid_losses.append(loss_val.item())

            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            train_loss_modified = np.average(train_losses_modified)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_train_losses_modified.append(train_loss_modified)
            avg_valid_losses.append(valid_loss)

                    
                
            with torch.no_grad():
                ## El número de variables de entrada
                #n_vars = inputs.shape[1]
                epoch_len = len(str(n_epochs))

                ## Dependiendo del tipo se printea más o menos contenido
                if verbose==2:
                    if epoch % n_visualized ==0:
                        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}]')
                        pbar.set_description(print_msg,end=' ')
                        pbar.set_postfix('Train Loss %f, Train Loss Mod %f, Val Loss %f' %(float(train_loss),float(train_loss_modified),float(valid_loss)))
                        logger.info('%s,Train Loss %f, Train Loss Mod %f, Val Loss %f, ' %(print_msg,float(train_loss),float(train_loss_modified),float(valid_loss)))
                        for i in range(n_vars):
                            if i in var_mono_crec:
                                pbar.set_postfix({'Minimum Jacobian x_{}'.format(i+1): float(jacob[0,:, i].min())})
                            elif i in var_mono_decrec:
                                pbar.set_postfix({'Maximum Jacobian x_{}'.format(i+1): float(jacob[0,:, i].max())})
                            else:
                                pbar.set_postfix({'Min Jacobian x_{}'.format(i+1): float(jacob[0,:, i].min()), 'Max Jacobian x_{}'.format(i+1): float(jacob[0,:, i].max())})
                elif verbose==1:
                    print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}]')
                    pbar.set_description(print_msg)
                    pbar.set_postfix({'Train Loss': float(train_loss), 'Train Loss Mod': float(train_loss_modified), 'Val Loss': float(valid_loss)})
                    logger.info('%s,Train Loss %f, Train Loss Mod %f, Val Loss %f, ' %(print_msg,float(train_loss),float(train_loss_modified),float(valid_loss)))
                else:
                    if epoch == n_epochs:
                        ## Only print in last epoch
                        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}]')
                        pbar.set_description(print_msg)
                        pbar.set_postfix({'Train Loss': float(train_loss), 'Train Loss Mod': float(train_loss_modified), 'Val Loss': float(valid_loss)})
                        logger.info('%s,Train Loss %f, Train Loss Mod %f, Val Loss %f, ' %(print_msg,float(train_loss),float(train_loss_modified),float(valid_loss)))
            
            # clear lists to track next epoch
            train_losses = []
            train_losses_modified = []
            valid_losses = []

            
            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, self._model)

            if early_stopping.early_stop:
                print("Early stopping at epoch %d" %(epoch))
                break
        
        # load the last checkpoint with the best model
        self._model.load_state_dict(torch.load(path))
        ## Delete model in model path if keep_model is False
        if not keep_model:
            os.remove(path)
        
        

        ## Almacenamos las curvas de entrenamiento
        self._avg_train_losses = avg_train_losses
        self._avg_train_losses_modified = avg_train_losses_modified
        self._avg_valid_losses = avg_valid_losses

        ## Almacenado en la propia clase
        #if delta>0:
        #    self._jacobian = jacob
        
    def plot_history(self,figsize=(10,5)):
        """
        This method generates a plot of the training and validation loss values of the neural network model over the epochs during training.
        The plot also includes a vertical line indicating the epoch with the lowest validation loss, which can be used as an early stopping checkpoint.

        Returns:
            None

        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        ax1.plot(range(1,len(self._avg_train_losses)+1),self._avg_train_losses, label='Training Loss')
        ax1.plot(range(1,len(self._avg_valid_losses)+1),self._avg_valid_losses,label='Validation Loss')

        # find position of lowest validation loss
        minposs = self._avg_valid_losses.index(min(self._avg_valid_losses))+1 
        ax1.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

        ax1.set_xlabel('epochs')
        ax1.set_ylabel('loss (log)')
        ax1.set_yscale('log')
        ax1.set_xlim(0, len(self._avg_train_losses)+1) # consistent scale
        ax1.grid(True)
        ax1.legend()

        diferences = [a_i - b_i for a_i, b_i in zip(self._avg_train_losses_modified, self._avg_train_losses)]
        print(len(diferences))
        ax2.plot(range(1,len(self._avg_train_losses_modified)+1),diferences, label='Modified Training Loss')
        ax2.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('loss (log)')
        ax2.set_yscale('log')
        ax2.set_xlim(0, len(self._avg_train_losses_modified)+1) # consistent scale
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.show()


    
    def closure_non_monotonic(self,inputs, targets, criterion, optimizer):
        """Closure function for optimizing the neural network's parameters using a given optimizer and loss criterion.
        -> Esta es válida para el caso de MSE
        Args:
            inputs (torch.Tensor): Input tensor for the neural network.
            targets (torch.Tensor): Target tensor for the neural network.
            criterion (torch.nn.Module): Loss criterion used to optimize the neural network.
            optimizer (torch.optim.Optimizer): Optimizer used to update the neural network parameters.

        Returns:
            loss (float): The loss value after backpropagation.

        """
        optimizer.zero_grad()
        outputs = self._model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        return loss
    
    def predict(self, inputs):
        """Realiza predicciones del modelo

        Args:
            inputs (torch.tensor): Data input

        Returns:
            numpy.array: Predicciones almacenadas en un array de numpy
        """
        self._model.eval()
        with torch.no_grad():
            predictions = self._model(inputs)
        return predictions
    
    def _plot_results(self,n_columna,X_tensor,Y_tensor):
        """Dibuja el output con respecto a la variable x(n_columna)

        Args:
            n_columna (_type_): _description_
            inputs (_type_): _description_
        """
        print('------------------ Representación gráfica ------------------')
        plt.scatter(X_tensor.numpy()[:,n_columna],Y_tensor.numpy().reshape(-1),color='blue',label='True')
        plt.scatter(X_tensor.numpy()[:,n_columna],self.predict(X_tensor).reshape(-1),label='Pred',color='orange')
        plt.legend()
        plt.show()

    def _plot_surface(self,X_tensor,n_var_1,n_var_2):

        # Obtenemos los valores máximos y mínimos de cada variable
        x1_min, _ = torch.min(X_tensor[:, n_var_1], dim=0)
        x1_max, _ = torch.max(X_tensor[:, n_var_1], dim=0)
        x2_min, _ = torch.min(X_tensor[:, n_var_2], dim=0)
        x2_max, _ = torch.max(X_tensor[:, n_var_2], dim=0)

        # Creamos un grid en dos dimensiones
        x1_grid, x2_grid = torch.meshgrid(
            torch.linspace(x1_min, x1_max, 100),  # 100 puntos entre x1_min y x1_max
            torch.linspace(x2_min, x2_max, 100)   # 100 puntos entre x2_min y x2_max
        )
        # Aplanamos los valores de x1_grid y x2_grid para tener dos tensores de tamaño (n_samples, 1) cada uno
        x1_flat = x1_grid.flatten().unsqueeze(1)
        x2_flat = x2_grid.flatten().unsqueeze(1)

        # Concatenamos x1_flat y x2_flat a lo largo de la dimensión 1 para obtener un tensor de entrada de tamaño (n_samples, 2)
        entrada_grid = torch.cat((x1_flat, x2_flat), dim=1)

        # Utilizamos el modelo para predecir los valores en cada punto del grid
        salida_grid = self._model(entrada_grid)

        # Graficamos la superficie resultante

        ax = plt.axes(projection='3d')
        ax.plot_surface(x1_grid.numpy(),x2_grid.numpy(), salida_grid.reshape(x1_grid.shape).detach().numpy(),)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('y')
        ax.set_title('Surface')
        plt.show()

    def _plot_surface_interactive(self,X_tensor,y_tensor,n_var_1,n_var_2):

        # Obtenemos los valores máximos y mínimos de cada variable
        x1_min, _ = torch.min(X_tensor[:, n_var_1], dim=0)
        x1_max, _ = torch.max(X_tensor[:, n_var_1], dim=0)
        x2_min, _ = torch.min(X_tensor[:, n_var_2], dim=0)
        x2_max, _ = torch.max(X_tensor[:, n_var_2], dim=0)

        # Crear el grid de datos
        x1, x2 = torch.meshgrid(torch.linspace(x1_min, x1_max, 100), torch.linspace(x2_min, x2_max, 100))

        # Hacer las predicciones para cada punto del grid
        X_grid = torch.stack([x1.reshape(-1), x2.reshape(-1)], axis=1)
        y_grid = self._model(X_grid).reshape(x1.shape).detach().numpy()

        # Crear la figura de Plotly
        fig = go.Figure()

        # Agregar la superficie
        fig.add_trace(go.Surface(x=x1, y=x2, z=y_grid,opacity=0.5))

        # Agregar los puntos evaluados por self._model como esferas
        y_pred = self._model(X_tensor).detach().numpy().squeeze()
        colors = 'blue'
        fig.add_trace(go.Scatter3d(x=X_tensor[:, n_var_1], y=X_tensor[:, n_var_2], z=y_pred, mode='markers',
                                    marker=dict(
            size=5,
            color='blue',
            opacity=1.0,
            line=dict(
                color='black',
                width=0.5
            ),
            symbol='circle'
        ),
            name='Prediccion',opacity = 0.5
        ))
        ## Agregar los puntos reales
        fig.add_trace(go.Scatter3d(x=X_tensor[:, n_var_1], y=X_tensor[:, n_var_2], z=y_tensor[:,0], mode='markers',
                                    marker=dict(
            size=5,
            color='orange',
            opacity=1.0,
            line=dict(
                color='black',
                width=0.5
            ),
            symbol='circle'
        ),
        name='Valores reales de entrenamiento'
    ))
        # Personalizar la figura
        fig.update_layout(
            title='Predicción de la superficie para el modelo MLP',
            width=800,
            height=1000,
            legend=dict(
                title=dict(text='Leyenda'),
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="white",
                bordercolor="Black",
                borderwidth=2
            ),
            scene=dict(
                xaxis_title='x1',
                yaxis_title='x2',
                zaxis_title='y',
                xaxis=dict(range=[x1_min, x1_max]),
                yaxis=dict(range=[x2_min, x2_max]),
                aspectmode='cube',
                aspectratio=dict(x=1, y=1, z=0.5),
                camera_eye=dict(x=-1.5, y=-1.5, z=0.5)
            )
        )
        

        # Mostrar la figura
        fig.show()
    
    
    def _plot_surface_interactive_proyected(self,X_tensor,y_tensor,n_var_1,n_var_2):
        """ The function _plot_surface_interactive_proyected generates an interactive 3D plot with a surface and two sets of points for a trained MLP model. The surface is generated from a grid where only two variables vary while the rest are taken as the mean (projection on the median in the space).

        Args:

        X_tensor: A PyTorch tensor with the input features to the model.
        y_tensor: A PyTorch tensor with the output values of the model.
        n_var_1: An integer indicating the index of the first variable to vary in the grid.
        n_var_2: An integer indicating the index of the second variable to vary in the grid.
        Returns:

        None.
        Note:
        This function assumes that the MLP model has already been trained and stored in the instance 
        variable _model. The function uses the _model to predict the output values for the 
        grid and the input features to generate the surface and the points. The function also
        requires the plotly package to create the interactive 3D plot.
        """

        ## Se toman el numero de variables
        n_variables = X_tensor.shape[1]

        # Obtenemos los valores máximos y mínimos de cada variable
        var_mins = []
        var_maxs = []

        ## Como vamos a hacer la proyeccion sobre la media en las variables no seleccionadas
        ## cambiamos los valores en esas coordenadas por la media (Como hacer un corte en el espacio por el valor medio)
        for i in range(n_variables):
            if i == n_var_1 or i == n_var_2:
                var_min, _ = torch.min(X_tensor[:, i], dim=0)
                var_max, _ = torch.max(X_tensor[:, i], dim=0)
            else:
                ## Se añade como valor maximo y minimo la media
                var_min = torch.median(X_tensor[:, i])
                var_max = torch.median(X_tensor[:, i])

                #Se convierte el valor del tensor a esto ya que luego se usará para la prediccion
                X_tensor[:, i] = torch.tensor(torch.median(X_tensor[:, i])).repeat(X_tensor[:, i].shape[0]).clone().detach()
            var_mins.append(var_min)
            var_maxs.append(var_max)

        ## Generamos lo que luego será el contenido de los grids
        grids = []

        for i in range(n_variables):
            grid = torch.linspace(var_mins[i], var_maxs[i], 100)
            grids.append(grid)

        meshgrids = torch.meshgrid(*grids)
        #entrada_grid = torch.cat([grid.reshape(-1, 1) for grid in meshgrids], dim=1)
        X_grid = torch.stack([x.reshape(-1) for x in meshgrids], axis=1)
        y_grid = self._model(X_grid).reshape(meshgrids[0].shape).detach().numpy()

        ## Debemos cortar el array multidimensional por las variables que se quieren representar
        ## (En realidad se trata de un grid que varian los valores en n_var_1 y n_var_2 mientras que en el resto de coordenadas es como si tomaramos cortes de un cubo)

        indices=[]
        for i in range(n_variables):
            if i==n_var_1 or i==n_var_2:
                indices.append(':')
            else:
                indices.append('0')
        ## Se construye el string que toma los cortes (por ejemplo: array[:,0,:])
        ## para que tenga tamaño (100,100) (en vex de (100,100,100) si es tridimensional)
        string =','.join(indices)
        formatted_string = f'({string.replace(":", "slice(None)")})'
        indices = eval(formatted_string)
        array_filtered = y_grid[indices]

        ## Reducimos tambien los meshgrids
        meshgrids_filtered_var_1 = meshgrids[n_var_1][indices]
        meshgrids_filtered_var_2 = meshgrids[n_var_2][indices]

        # Crear la figura de Plotly
        fig = go.Figure()

        # Agregar la superficie
        fig.add_trace(go.Surface(x=meshgrids_filtered_var_1, y=meshgrids_filtered_var_2, z=array_filtered,opacity=0.5))

        # Agregar los puntos evaluados por self._model como esferas
        y_pred = self._model(X_tensor).detach().numpy().squeeze()
        fig.add_trace(go.Scatter3d(x=X_tensor[:, n_var_1], y=X_tensor[:, n_var_2], z=y_pred, mode='markers',
                                    marker=dict(
            size=5,
            color='blue',
            opacity=1.0,
            line=dict(
                color='black',
                width=0.5
            ),
            symbol='circle'
        ),
            name='Prediccion'
        ))
        ## Agregar los puntos reales
        fig.add_trace(go.Scatter3d(x=X_tensor[:, n_var_1], y=X_tensor[:, n_var_2], z=y_tensor[:,0], mode='markers',
                                    marker=dict(
            size=5,
            color='orange',
            opacity=1.0,
            line=dict(
                color='black',
                width=0.5
            ),
            symbol='circle'
        ),
        name='Valores reales de entrenamiento'
    ))
        # Personalizar la figura
        fig.update_layout(
            title='Predicción de la superficie para el modelo MLP',
            legend=dict(
                title=dict(text='Leyenda'),
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="white",
                bordercolor="Black",
                borderwidth=2
            ),
            scene=dict(
                xaxis_title='x1',
                yaxis_title='x2',
                zaxis_title='y',
                xaxis=dict(range=[var_mins[n_var_1], var_maxs[n_var_1]]),
                yaxis=dict(range=[var_mins[n_var_2], var_maxs[n_var_2]]),
                aspectratio=dict(x=1, y=1, z=0.5),
                camera_eye=dict(x=-1.5, y=-1.5, z=0.5)
            )
        )
        

        # Mostrar la figura
        fig.show()

    def _plot_surface_interactive_3d(self, X_tensor, y_tensor, n_var_1, n_var_2, n_var_3):

        # Obtenemos los valores máximos y mínimos de cada variable
        x1_min, _ = torch.min(X_tensor[:, n_var_1], dim=0)
        x1_max, _ = torch.max(X_tensor[:, n_var_1], dim=0)
        x2_min, _ = torch.min(X_tensor[:, n_var_2], dim=0)
        x2_max, _ = torch.max(X_tensor[:, n_var_2], dim=0)

        # Crear el grid de datos para x1 y x2
        x1, x2 = torch.meshgrid(torch.linspace(x1_min, x1_max, 100), torch.linspace(x2_min, x2_max, 100))

        # Crear la figura de Plotly
        fig = go.Figure()

        # Crear un rango de valores para la tercera variable
        x3_range = torch.unique(X_tensor[:,n_var_3])

        # Para cada valor en el rango de la tercera variable, generar una traza de superficie
        for i, x3_value in enumerate(x3_range):
            # Hacer las predicciones para cada punto del grid
            X_grid = torch.stack([x1.reshape(-1), x2.reshape(-1), torch.full_like(x1.reshape(-1), x3_value)], axis=1)
            y_grid = self._model(X_grid).reshape(x1.shape).detach().numpy()

            # Agregar la superficie
            fig.add_trace(go.Surface(x=x1, y=x2, z=y_grid, opacity=0.5, visible=(i==0)))
            X_tensor_red = X_tensor[X_tensor[:, n_var_3] == x3_value][:,[n_var_1,n_var_2]]
            y_tensor_red = y_tensor[X_tensor[:, n_var_3] == x3_value]
            fig.add_trace(go.Scatter3d(x=X_tensor_red[:, n_var_1], y=X_tensor_red[:, n_var_2], z=y_tensor_red[:,0], mode='markers',
                                    marker=dict(
                        size=5,
                        color='orange',
                        opacity=1.0,
                        line=dict(
                            color='black',
                            width=0.5
                        ),
                        symbol='circle'
                    ),
                    name='Valores reales de entrenamiento',
                    visible=(i==0)
                ))
        # Crear un control deslizante para seleccionar el valor de la tercera variable
        steps = []
        for i, x3_value in enumerate(x3_range):
            step = dict(
                method="update",
                args=[{"visible": [j==i for j in range(len(x3_range))]}],
                label=f"{x3_value:.2f}"
            )
            steps.append(step)

        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Valor de la tercera variable: "},
            pad={"t": 50},
            steps=steps
        )]

        # Personalizar la figura
        fig.update_layout(
            sliders=sliders,
            title='Predicción de la superficie para el modelo MLP',
            width=800,
            height=1000,
            scene=dict(
                xaxis_title='x1',
                yaxis_title='x2',
                zaxis_title='y',
                xaxis=dict(range=[x1_min, x1_max]),
                yaxis=dict(range=[x2_min, x2_max]),
                zaxis=dict(range = [0,1]),
                aspectmode='cube',
                aspectratio=dict(x=1, y=1, z=0.5),
                camera_eye=dict(x=-1.5, y=-1.5, z=0.5)
            )
        )

        # Mostrar la figura
        fig.show()



    def _plot_jacobian_interactive(self,X_tensor,n_var_1,n_var_2,var_rep):
        """_summary_

        Args:
            X_tensor (_type_): _description_
            n_var_1 (_type_): _description_
            n_var_2 (_type_): _description_
            var_rep (_type_): Variable sobre la que se representa la salida del jacobiano (Que parciales queremos mostrar)

        """

        # Obtenemos los valores máximos y mínimos de cada variable
        x1_min, _ = torch.min(X_tensor[:, n_var_1], dim=0)
        x1_max, _ = torch.max(X_tensor[:, n_var_1], dim=0)
        x2_min, _ = torch.min(X_tensor[:, n_var_2], dim=0)
        x2_max, _ = torch.max(X_tensor[:, n_var_2], dim=0)

        # Crear el grid de datos
        x1, x2 = torch.meshgrid(torch.linspace(x1_min, x1_max, 100), torch.linspace(x2_min, x2_max, 100))

        # Hacer las predicciones para cada punto del grid
        X_grid = torch.stack([x1.reshape(-1), x2.reshape(-1)], axis=1)
        y_grid = self.batch_jacobian(X_grid)
        y_grid = y_grid[0,:,var_rep].reshape(x1.shape).detach().numpy()

        # Crear la figura de Plotly
        fig2 = go.Figure()

        # Agregar la superficie
        fig2.add_trace(go.Surface(x=x1, y=x2, z=y_grid,opacity=0.5))
        # Agregar los puntos evaluados por self._model como esferas
        y_pred = self.batch_jacobian(X_tensor)
        y_pred_plot = y_pred[0,:,var_rep].reshape(X_tensor[:, n_var_1].shape).detach().numpy()
        fig2.add_trace(go.Scatter3d(x=X_tensor[:, n_var_1], y=X_tensor[:, n_var_2], z=y_pred_plot, mode='markers',
                                    marker=dict(
            size=5,
            color='blue',
            opacity=1.0,
            line=dict(
                color='black',
                width=0.5
            ),
            symbol='circle'
        ),
            name='Prediccion'
        ))
        # Agregar el plano en z=0
        z_plane = np.zeros_like(y_grid)  # Create a plane at z=0
        fig2.add_trace(go.Surface(x=x1, y=x2, z=z_plane, opacity=0.5, showscale=False))
    
        # Personalizar la figura
        fig2.update_layout(
            title='Predicción de la superficie del jacobiano para el modelo MLP',
            legend=dict(
                title=dict(text='Leyenda'),
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="white",
                bordercolor="Black",
                borderwidth=2
            ),
            scene=dict(
                xaxis_title='x1',
                yaxis_title='x2',
                zaxis_title='y',
                xaxis=dict(range=[x1_min, x1_max]),
                yaxis=dict(range=[x2_min, x2_max]),
                aspectratio=dict(x=1, y=1, z=0.5),
                camera_eye=dict(x=-1.5, y=-1.5, z=0.5)
            ),
            width=1000,  # Cambia el ancho de la figura
            height=800
        )

        # Mostrar la figura
        fig2.show()
    
    def _plot_jacobian_interactive_3d(self, X_tensor, y_tensor, n_var_1, n_var_2, n_var_3, var_rep):

        # Get the minimum and maximum values of each variable
        x1_min, _ = torch.min(X_tensor[:, n_var_1], dim=0)
        x1_max, _ = torch.max(X_tensor[:, n_var_1], dim=0)
        x2_min, _ = torch.min(X_tensor[:, n_var_2], dim=0)
        x2_max, _ = torch.max(X_tensor[:, n_var_2], dim=0)

        # Create the data grid for x1 and x2
        x1, x2 = torch.meshgrid(torch.linspace(x1_min, x1_max, 100), torch.linspace(x2_min, x2_max, 100))

        # Create the Plotly figure
        fig = go.Figure()

        # Create a range of values for the third variable
        x3_range = torch.unique(X_tensor[:,n_var_3])

        # For each value in the range of the third variable, generate a surface trace
        for i, x3_value in enumerate(x3_range):
            # Make predictions for each point on the grid
            X_grid = torch.stack([x1.reshape(-1), x2.reshape(-1), torch.full_like(x1.reshape(-1), x3_value)], axis=1)
            y_grid = self.batch_jacobian(X_grid)
            y_grid = y_grid[0,:,var_rep].reshape(x1.shape).detach().numpy()

            # Add the surface
            fig.add_trace(go.Surface(x=x1, y=x2, z=y_grid, opacity=0.5, visible=(i==0)))
            """ X_tensor_red = X_tensor[X_tensor[:, n_var_3] == x3_value][:,[n_var_1,n_var_2]]
            y_tensor_red = y_tensor[X_tensor[:, n_var_3] == x3_value]
            fig.add_trace(go.Scatter3d(x=X_tensor_red[:, n_var_1], y=X_tensor_red[:, n_var_2], z=y_tensor_red[:,0], mode='markers',
                                    marker=dict(
                        size=5,
                        color='orange',
                        opacity=1.0,
                        line=dict(
                            color='black',
                            width=0.5
                        ),
                        symbol='circle'
                    ),
                    name='Real training values',
                    visible=(i==0)
                ))"""
            
            # Agregar el plano en z=0
            z_plane = np.zeros_like(y_grid)  # Create a plane at z=0
            fig.add_trace(go.Surface(x=x1, y=x2, z=z_plane, opacity=0.5, showscale=False))

        # Create a slider to select the value of the third variable
        steps = []
        for i, x3_value in enumerate(x3_range):
            step = dict(
                method="update",
                args=[{"visible": [j==i for j in range(len(x3_range))]}],
                label=f"{x3_value:.2f}"
            )
            steps.append(step)

        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Value of the third variable: "},
            pad={"t": 50},
            steps=steps
        )]

        # Customize the figure
        fig.update_layout(
            sliders=sliders,
            title='Prediction of the Jacobian surface for the MLP model',
            width=800,
            height=1000,
            scene=dict(
                xaxis_title='x1',
                yaxis_title='x2',
                zaxis_title='y',
                xaxis=dict(range=[x1_min, x1_max]),
                yaxis=dict(range=[x2_min, x2_max]),
                zaxis=dict(range = [0,1]),
                aspectmode='cube',
                aspectratio=dict(x=1, y=1, z=0.5),
                camera_eye=dict(x=-1.5, y=-1.5, z=0.5)
            )
        )

        # Show the figure
        fig.show()
   

    def _plot_jacobian_interactive_proyected(self,X_tensor,n_var_1,n_var_2,var_rep):
        """_summary_

        Args:
            X_tensor (_type_): _description_
            n_var_1 (_type_): _description_
            n_var_2 (_type_): _description_
            var_rep (_type_): Variable sobre la que se representa la salida del jacobiano
        """

        ## Se toman el numero de variables
        n_variables = X_tensor.shape[1]

        # Obtenemos los valores máximos y mínimos de cada variable
        var_mins = []
        var_maxs = []

        ## Como vamos a hacer la proyeccion sobre la media en las variables no seleccionadas
        ## cambiamos los valores en esas coordenadas por la media (Como hacer un corte en el espacio por el valor medio)
        for i in range(n_variables):
            if i == n_var_1 or i == n_var_2:
                var_min, _ = torch.min(X_tensor[:, i], dim=0)
                var_max, _ = torch.max(X_tensor[:, i], dim=0)
            else:
                ## Se añade como valor maximo y minimo la media
                var_min = torch.median(X_tensor[:, i])
                var_max = torch.median(X_tensor[:, i])
                #Se convierte el valor del tensor a esto ya que luego se usará para la prediccion
                X_tensor[:, i] = torch.tensor(torch.median(X_tensor[:, i])).repeat(X_tensor[:, 2].shape[0])
            var_mins.append(var_min)
            var_maxs.append(var_max)

        ## Generamos lo que luego será el contenido de los grids
        grids = []

        for i in range(n_variables):
            grid = torch.linspace(var_mins[i], var_maxs[i], 100)
            grids.append(grid)

        meshgrids = torch.meshgrid(*grids)
        #entrada_grid = torch.cat([grid.reshape(-1, 1) for grid in meshgrids], dim=1)
        X_grid = torch.stack([x.reshape(-1) for x in meshgrids], axis=1)
        y_grid = self.batch_jacobian(X_grid)
        y_grid_proyected = y_grid[0,:,var_rep].reshape(meshgrids[0].shape).detach().numpy()

        ## Debemos cortar el array multidimensional por las variables que se quieren representar
        ## (En realidad se trata de un grid que varian los valores en n_var_1 y n_var_2 mientras que en el resto de coordenadas es como si tomaramos cortes de un cubo)

        indices=[]
        for i in range(n_variables):
            if i==n_var_1 or i==n_var_2:
                indices.append(':')
            else:
                indices.append('0')
        ## Se construye el string que toma los cortes (por ejemplo: array[:,0,:])
        ## para que tenga tamaño (100,100) (en vex de (100,100,100) si es tridimensional)
        string =','.join(indices)
        formatted_string = f'({string.replace(":", "slice(None)")})'
        indices = eval(formatted_string)
        array_filtered = y_grid_proyected[indices]

        ## Reducimos tambien los meshgrids
        meshgrids_filtered_var_1 = meshgrids[n_var_1][indices]
        meshgrids_filtered_var_2 = meshgrids[n_var_2][indices]
        # Crear la figura de Plotly
        fig2 = go.Figure()

        # Agregar la superficie
        fig2.add_trace(go.Surface(x=meshgrids_filtered_var_1, y=meshgrids_filtered_var_2, z=array_filtered,opacity=0.5))
        # Agregar los puntos evaluados por self._model como esferas
        y_pred = self.batch_jacobian(X_tensor)
        y_pred_plot = y_pred[0,:,var_rep].reshape(X_tensor[:, n_var_1].shape).detach().numpy()
        fig2.add_trace(go.Scatter3d(x=X_tensor[:, n_var_1], y=X_tensor[:, n_var_2], z=y_pred_plot, mode='markers',
                                    marker=dict(
            size=5,
            color='blue',
            opacity=1.0,
            line=dict(
                color='black',
                width=0.5
            ),
            symbol='circle'
        ),
            name='Prediccion'
        ))
         # Agregar el plano en z=0
        z_plane = np.zeros_like(y_grid)  # Create a plane at z=0
        fig2.add_trace(go.Surface(x=meshgrids_filtered_var_1, y=meshgrids_filtered_var_2, z=z_plane, opacity=0.5, showscale=False))

        # Personalizar la figura
        fig2.update_layout(
            title='Predicción de la superficie del jacobiano para el modelo MLP',
            legend=dict(
                title=dict(text='Leyenda'),
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="white",
                bordercolor="Black",
                borderwidth=2
            ),
            scene=dict(
                xaxis_title='x1',
                yaxis_title='x2',
                zaxis_title='y',
                xaxis=dict(range=[var_mins[n_var_1], var_maxs[n_var_1]]),
                yaxis=dict(range=[var_mins[n_var_2], var_maxs[n_var_2]]),
                aspectratio=dict(x=1, y=1, z=0.5),
                camera_eye=dict(x=-1.5, y=-1.5, z=0.5)
            )
        )

        # Mostrar la figura
        fig2.show()