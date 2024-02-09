from datetime import datetime
import logging
import os
import yaml

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


## Add Utilis related with Monotonic Training
import sys
## Windows Version
if sys.platform == 'win32':
    sys.path.insert(0, 'C:/Users/apolo/OneDrive - Universidad Pontificia Comillas/Escritorio/PhD/Codigo/NNMonotonic/Scripts/')
    sys.path.insert(0, 'C:/Users/apolo/OneDrive - Universidad Pontificia Comillas/Escritorio/IIT/NN Monotonic/Scripts/')
## Mac Version
if sys.platform == 'darwin':
    sys.path.insert(0,'/Users/alejandropolo/Library/CloudStorage/OneDrive-UniversidadPontificiaComillas/Escritorio/PhD/Codigo/LipschitzNN/Scripts/')
    sys.path.insert(0,'/Users/alejandropolo/Library/CloudStorage/OneDrive-UniversidadPontificiaComillas/Escritorio/PhD/Codigo/LipschitzNN/')
    sys.path.insert(0,'/Users/alejandropolo/Library/CloudStorage/OneDrive-UniversidadPontificiaComillas/Escritorio/IIT/NN Monotonic/Scripts/')

import MLP_Monotonic
import Utilities as Utilities
from DNN import DNN
import utilities_voronoi
import importlib

from utilities_voronoi import *
importlib.reload(utilities_voronoi)
importlib.reload(MLP_Monotonic)
importlib.reload(Utilities)

from MLP_Monotonic import MLP_Monotonic


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
logger.setLevel(logging.DEBUG)


### CARGAR EL YAML DE CONFIGURACIÓN
logger.info(f"-------------------Cargando el YAMl de configuración-------------------")
path = '../Scripts/config.yaml'
with open(path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

## FUNCTION TO MODIFY THE YAML
def write_yaml(file_path, data):
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

def train(config):
    ############## LOAD TRAINING DATA ##################
    np.random.seed(config['training']['seed'])
    torch.manual_seed(config['training']['seed'])

    ### LOAD DATA
    X_train_tensor = torch.load('../Data/X_train_data.pt')
    X_test_tensor = torch.load('../Data/X_test_data.pt')
    y_train_tensor = torch.load('../Data/y_train_data.pt')
    y_test_tensor = torch.load('../Data/y_test_data.pt')


    ## Convert to pytorch data load
    n_samples = len(X_train_tensor)
    train_dt = TensorDataset(X_train_tensor,y_train_tensor) # create your datset
    train_dataload = DataLoader(train_dt,batch_size=64) # create your dataloader

    ## Convert to pytorch data load
    n_samples = len(X_test_tensor)
    val_dt = TensorDataset(X_test_tensor,y_test_tensor) # create your datset
    val_dataload = DataLoader(val_dt,batch_size=n_samples) # create your dataloader

    ############## MODEL TRAINING ##################
    ### Define the model
    model = DNN(config['model_architecture']['layers'],activations=config['model_architecture']['actfunc'])
    
    ## Add the initial and final activation functions
    ## Check if first and last activation functions are 'identity'
    if config['model_architecture']['actfunc'][0] != 'identity':
        config['model_architecture']['actfunc'].insert(0,'identity')
    if config['model_architecture']['actfunc'][-1] != 'identity':
        config['model_architecture']['actfunc'].append('identity')
    ## Add the initial and final activation functions
    #write_yaml(path, config)

    ## Define the loss function
    criterion = nn.MSELoss()

    ## Start the class of the model
    mlp_model = MLP_Monotonic(_model_name="Prueba",_model = model)
    print('------------------ Training ------------------')

    if config['training']['epsilon'] is not None:
        eps = config['training']['epsilon']
    else:
        eps = 0.0

    mlp_model.train_adjusted_std(train_data=train_dataload,val_data=val_dataload,criterion=criterion,epsilon = eps,
                                    n_epochs=config['training']['n_epochs'],categorical_columns=[],verbose=config['training']['verbose'],n_visualized=1,
                                    monotone_relations=config['training']['monotone_relations'],optimizer_type=config['training']['optimizer_type'],
                                    learning_rate=config['training']['learning_rate'],weight_decay=config['training']['weight_decay'],
                                    delta=config['training']['delta'],patience=config['training']['patience'],
                                    delta_synthetic=config['training']['delta_synthetic'],delta_external=config['training']['delta_external'],
                                    std_growth=config['training']['std_growth'],epsilon_synthetic=config['training']['epsilon_synthetic'],
                                    model_path='./Models/checkpoint_mlp_',external_points=None,seed=2023)

    print('------------------ Training Results ------------------')
    ### Ploteado de resultados
    if config['training']['plot_history']:
        mlp_model.plot_history()

    ## Save the model with the name NN+timestamp
    if config['training']['save_model']:
        mlp_model.save_model()
        

    return mlp_model

if __name__ == "__main__":
    logger.info(f"-------------------Iniciando el entrenamiento-------------------")
    train(config)
    logger.info(f"-------------------Entrenamiento finalizado-------------------")