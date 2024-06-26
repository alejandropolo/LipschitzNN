################# LOAD LIBRARIES ####################
import torch
import pandas as pd
import numpy as np
import os
import time
from sklearn.metrics import r2_score

def print_errors(model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor,lipschitz_const,log=False,config=None):

    device = next(model.parameters()).device 

    X_train_tensor = X_train_tensor.to(device)
    X_test_tensor = X_test_tensor.to(device)
    X_val_tensor = X_val_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)
    y_test_tensor = y_test_tensor.to(device)
    y_val_tensor = y_val_tensor.to(device)


    y_train_pred = model(X_train_tensor)
    y_test_pred = model(X_test_tensor)
    y_val_pred = model(X_val_tensor)
    mse_train =  torch.mean((y_train_tensor - y_train_pred)**2)  
    mse_test =  torch.mean((y_test_tensor - y_test_pred)**2)  
    mse_val =  torch.mean((y_val_tensor - y_val_pred)**2)
    rmse_train =  torch.sqrt(mse_train)  
    rmse_test =  torch.sqrt(mse_test)
    rmse_val =  torch.sqrt(mse_val)  
    mae_train =  torch.mean(torch.abs(y_train_tensor - y_train_pred))  
    mae_test =  torch.mean(torch.abs(y_test_tensor - y_test_pred))
    mae_val =  torch.mean(torch.abs(y_val_tensor - y_val_pred)) 
    r2_train = r2_score(y_train_tensor.detach().cpu().numpy(), y_train_pred.detach().cpu().numpy()) 
    r2_test = r2_score(y_test_tensor.detach().cpu().numpy(), y_test_pred.detach().cpu().numpy())
    r2_val = r2_score(y_val_tensor.detach().cpu().numpy(), y_val_pred.detach().cpu().numpy())

    results = pd.DataFrame(columns=["Timestamp", "Layers", "activations","Epochs",
                                    "delta",
                                    "lr","weight_decay","lipschitz_const",
                                    "RMSE_Train","RMSE_Val", "RMSE_Test", "MAE_Train", "MAE_Val","MAE_Test","R2_Train","R2_Val","R2_Test"])

    if log and not config is None:
        log_path = "../logs/errors_log.csv"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        new_row = pd.DataFrame({
            "Timestamp": time.time(),
            "Layers": str(list(config['model_architecture']['layers'])),
            "activations": str(list(config['model_architecture']['actfunc'])),
            "Epochs": config['training']['n_epochs'],
            "delta": config['training']['delta'],
            "lr": config['training']['learning_rate'],
            "weight_decay": config['training']['weight_decay'],
            "lipschitz_const": lipschitz_const,
            "RMSE_Train": np.round(rmse_train.item(),6),
            "RMSE_Val": np.round(rmse_val.item(),6),
            "RMSE_Test": np.round(rmse_test.item(),6),
            "MAE_Train": np.round(mae_train.item(),6),
            "MAE_Val": np.round(mae_val.item(),6),
            "MAE_Test": np.round(mae_test.item(),6),
            "R2_Train": np.round(r2_train,6),
            "R2_Val": np.round(r2_val,6),
            "R2_Test": np.round(r2_test,6)
        }, index=[0])

        results = pd.concat([results.dropna(), new_row], ignore_index=True)

        results.to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)

    print(f"MSE Train: {np.round(mse_train.item(),5)}, MSE Val: {np.round(mse_val.item(),5)}, MSE Test: {np.round(mse_test.item(),5)}")
    print(f"RMSE Train: {np.round(rmse_train.item(),5)}, RMSE Val: {np.round(rmse_val.item(),5)} , RMSE Test: {np.round(rmse_test.item(),5)}")
    print(f"MAE Train: {np.round(mae_train.item(),5)}, MAE Val: {np.round(mae_val.item(),5)}, MAE Test: {np.round(mae_test.item(),5)}")
    print(f"R2 Train: {np.round(r2_train,5)}, R2 Val: {np.round(r2_val,5)}, R2 Test: {np.round(r2_test,5)}")
    