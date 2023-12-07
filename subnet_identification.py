import deepSI
from torch import nn
import os
from utils import *
from matplotlib import pyplot as plt

# Hyperparameters
T = 50
epochs = 100
batch_size = 512
hl_enc = 3
n_nodes_enc = 64
hl_din = 2
n_nodes_din = 64
activation = nn.Tanh
nx = 16
all_lag = 30
na = all_lag
nb = all_lag
nu = 1
ny = 1
learning_rate = 1e-3

# Data concatenation:
TrainDataFolderName = "SimData/Train"
TestDataFolderName = "SimData/Test"
ValidDataFolderName = "SimData/Valid"
CurrDir = os.getcwd()
train_data_folders = os.path.join(CurrDir, TrainDataFolderName)
valid_data_folders = os.path.join(CurrDir, ValidDataFolderName)
test_data_folders = os.path.join(CurrDir, TestDataFolderName)

train = create_sysdata_from_file(train_data_folders)
valid = create_sysdata_from_file(valid_data_folders)
testdata = create_sysdata_from_file(test_data_folders)

# Initialization:
e_net = deepSI.fit_systems.encoders.default_encoder_net
f_net = deepSI.fit_systems.encoders.default_state_net
h_net = deepSI.fit_systems.encoders.default_output_net

fitsys = deepSI.fit_systems.SS_encoder_general(nx=nx, na=na, nb=nb, e_net=e_net, f_net=f_net, h_net=h_net,
                                               e_net_kwargs=dict(n_nodes_per_layer=n_nodes_enc, n_hidden_layers=hl_enc, activation=activation),
                                               f_net_kwargs=dict(n_nodes_per_layer=n_nodes_din, n_hidden_layers=hl_din, activation=activation),
                                               h_net_kwargs=dict(n_nodes_per_layer=n_nodes_din, n_hidden_layers=hl_din, activation=activation))

# Training:
fitsys.fit(train_sys_data=train, val_sys_data=valid, epochs=epochs, batch_size=batch_size, loss_kwargs=dict(nf=T),
           auto_fit_norm=True, optimizer_kwargs=dict(lr=learning_rate), validation_measure='30-step-NRMS')  #ToDo: ask for prediction hotizon

# Training losses
fitsys.checkpoint_load_system(name='_last')
train_losses = fitsys.Loss_train[:]
val_losses = fitsys.Loss_val[:]

fig_losses, axs = plt.subplots(2, 1, sharex=True)

axs[0].semilogy(train_losses, label='Training loss')
axs[0].legend()
axs[0].set_ylabel('MS')

axs[1].semilogy(val_losses, label='Validation loss')
axs[1].legend()
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('NRMS')
axs[1].legend()

fig_losses.tight_layout()
plt.show()

# Testing:
fitsys.checkpoint_load_system(name='_best')
valid_results = fitsys.apply_experiment(valid, save_state=False)
test_results = fitsys.apply_experiment(testdata, save_state=False)

RMS_valid = valid_results.RMS(valid)
NRMS_valid = valid_results.NRMS(valid)
RMS_test = test_results.RMS(testdata)
NRMS_test = test_results.NRMS(testdata)
print(f'Validation NRMS error: {NRMS_valid:.2%}')
print(f'Test NRMS error: {NRMS_test:.2%}')

# Saving:
encoder_data = {
    "Truncation length": T,
    "Epochs": epochs,
    "Batch size": batch_size,
    "Hidden layers (h+f)": hl_din,
    "Nodes per layer (h+f)": n_nodes_din,
    "n_hidden_layers_e": hl_enc,
    "n_nodes_per_layer_e": n_nodes_enc,
    "activation function": str(activation),
    "nf": all_lag,
    "ny": ny,
    "nu": nu,
    "nx": nx,
    "Test RMS": RMS_test,
    "Test NRMS": NRMS_test,
    "Valid RMS": RMS_valid,
    "Valid NRMS": NRMS_valid
}

save_encoder(fitsys, encoder_data, fig_losses)
