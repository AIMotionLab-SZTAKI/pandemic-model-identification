import deepSI
from torch import nn
import os
from utils import *
from matplotlib import pyplot as plt


# SETTINGS
epochs = 500

# SUBNET path
cwd = os.getcwd()
folder_name = "2024_03_18_113943_OPnw3Q"
fit_sys = deepSI.load_system(os.path.join(cwd, 'EncoderData', folder_name, "best.pt"))
fit_sys.bestfit = 1e5

# Data concatenation:
train_data_folder = "SimData/Train"
valid_data_folder = "SimData/Valid"
test_data_folder = "SimData/Test"

train_data = create_sysdata_from_file(os.path.join(cwd, train_data_folder), 1)
valid_data = create_sysdata_from_file(os.path.join(cwd, valid_data_folder), 1)
test_data = create_sysdata_from_file(os.path.join(cwd, test_data_folder), 1)

refinement_folder = "SimData/Refinement"
refinement_data = create_refined_sysdata(os.path.join(cwd, refinement_folder))
separating_idx = int(0.8*len(refinement_data))
train_data.append(refinement_data[:separating_idx])
valid_data.append(refinement_data[separating_idx:])

# Training:
fit_sys.fit(train_sys_data=train_data, val_sys_data=valid_data, epochs=epochs, batch_size=256, loss_kwargs=dict(nf=60),
           auto_fit_norm=True, optimizer_kwargs=dict(lr=1e-3), validation_measure='7-step-NRMS')

# Training losses
fit_sys.checkpoint_load_system(name='_last')
train_losses = fit_sys.Loss_train[:]
val_losses = fit_sys.Loss_val[:]

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
fit_sys.checkpoint_load_system(name='_best')
valid_results = fit_sys.apply_experiment(valid_data, save_state=False)
test_results = fit_sys.apply_experiment(test_data, save_state=False)

RMS_valid = valid_results.RMS(valid_data)
NRMS_valid = valid_results.NRMS(valid_data)
RMS_test = test_results.RMS(test_data)
NRMS_test = test_results.NRMS(test_data)
print(f'Validation NRMS error: {NRMS_valid:.2%}')
print(f'Test NRMS error: {NRMS_test:.2%}')

# Saving:
save_encoder(fit_sys, fig_losses)
