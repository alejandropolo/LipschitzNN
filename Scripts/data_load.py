import numpy as np
import torch
from sklearn.model_selection import train_test_split
import yaml
import os 
### CARGAR EL YAML DE CONFIGURACIÓN
path = '../Scripts/config.yaml'
with open(path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

#### DATA LOAD FUNCTION

def data_load(config,f_output):

    #### DATA GENERATION
    noise = np.random.normal(0,config['data_load']['noise_constant'],config['data_load']['n_samples'])
    
    ### DATA GENERATION
    torch.manual_seed(2023)
    np.random.seed(0)

    # Crear una cuadrícula de valores que serviran como datos de train
    n_samples = 50
    x1_values = np.linspace(config['data_load']['x_lim'][0], config['data_load']['x_lim'][1], config['data_load']['n_samples'])
    x2_values = np.linspace(config['data_load']['y_lim'][0], config['data_load']['y_lim'][1], config['data_load']['n_samples'])
    x1_mesh, x2_mesh = np.meshgrid(x1_values, x2_values)
    ## Juntamos y convertimos a tensor
    mesh = np.hstack((x1_mesh.reshape(-1,1),x2_mesh.reshape(-1,1)))
    noise_matr=np.random.multivariate_normal(mean=[0,0], cov=[[0, 0], [0,0]], size=len(mesh))
    mesh = mesh+noise_matr
    input_mesh = torch.tensor(mesh).float()
    noise = np.random.normal(0,config['data_load']['noise_constant'],len(mesh))


    ## Convertimos los datos a un dataset para pytorch
    y=torch.tensor(f_output(mesh[:,0],mesh[:,1],noise)).reshape(len(mesh),1).float()

    ## Min max scaler to the output
    y_min = torch.min(y)
    y_max = torch.max(y)
    def min_max_scaler(y,y_min=y_min,y_max=y_max):
        return (y-y_min)/(y_max-y_min)
    y = min_max_scaler(y)

    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = train_test_split(input_mesh, y, test_size=0.25, random_state=2023)

    ## Save the data
    ## if ../Data folder does not exist, create it
    if not os.path.exists('../Data'):
        os.makedirs('../Data')
    torch.save(X_train_tensor,'../Data/X_train_data.pt')
    torch.save(X_test_tensor,'../Data/X_test_data.pt')
    torch.save(y_train_tensor,'../Data/y_train_data.pt')
    torch.save(y_test_tensor,'../Data/y_test_data.pt')


if __name__ == '__main__':
    data_load()