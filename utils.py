import deepSI
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pickle
import random

def csv_read(folder, path_parts=[]):
    """
    Reads the data into a .csv file.

    Arguments:
        - folder (string): path of the reading
        - path_parts (string): extra paths in the folder

    Returns:
        -data (np.array): data which been read from .csv file
    """

    for part in path_parts:
        folder = os.path.join(folder, part)
    with open(os.path.join(folder), 'r') as in_file:
        data = np.genfromtxt(in_file, delimiter=",")
    return data

def create_sysdata_from_file(data_folders):
    """
    Creates system (deepSI) data from time-series data.

    Arguments:
        - data_folders (string): path direction of the time-series data file
    Returns:
        - system_data (deepSI.system_data): training data
    """

    input_data = []
    output_data = []
    data_names_list = os.listdir(data_folders)
    for name in data_names_list:
        folder = os.path.join(data_folders, name)
        input_data.append(csv_read(folder, ['input.csv']))
        output_data.append(csv_read(folder, ['output.csv']))

    sys_data_list = []
    for input, output in zip(input_data, output_data):
            sys_data_part = deepSI.System_data(u=input, y=output)
            sys_data_list.append(sys_data_part)

    system_data = deepSI.System_data_list(sys_data_list=sys_data_list)
    return system_data

def save_encoder(fit_sys, encoder_data, fig_losses):
    """
    Saves the SUBNET-encoder.

    Arguments:
        - fit_sys (SUBNET): deepSI encoder
        - encoder_data (list): parameters of the encoder
        - fig_losses (figure): validation and taring loss during the training
        - type (string): encoder-type: continuous or discrete
        - name (string): data-type: simulation or measurement
    Returns: -
    """

    curr_dir = os.getcwd()
    data_name = "EncoderData"
    new_sys_folder = os.path.join(curr_dir, data_name,  datetime.now().strftime("%Y_%m_%d_%H%M%S") + '_' + fit_sys.unique_code)
    os.makedirs(new_sys_folder)

    # fit_sys.checkpoint_load_system(name='_last')
    # fit_sys.save_system(new_sys_folder + '/last.pt')
    fit_sys.checkpoint_load_system(name='_best')
    fit_sys.save_system(new_sys_folder + '/best.pt')

    fig_losses.savefig(new_sys_folder + '/losses.png', bbox_inches='tight')

    header = 'Training properties: \n'
    with open(new_sys_folder + '/info.txt', 'w') as f:
        f.write(header)
        for key, value in encoder_data.items():
            f.write('%s: %s\n' % (key, value))

    print("---------- Encoder saved ----------")

def plot_SISO_results(sim_results, test_data, blockfig=False):
    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(test_data.u)
    axs[0].set_ylabel('Input')

    axs[1].plot(test_data.y, label="PanSim")
    axs[1].plot(sim_results.y, label="SUBNET")
    axs[1].set_xlabel('Sim. index')
    axs[1].set_ylabel('Hospitalized people')
    axs[1].legend()

    fig.tight_layout()
    plt.show(block=blockfig)

def split_list(inputs, outputs, nf, T, split_fraction):
    """
    Splits a list(two) randomly in between and extract the remaining parts.

    Arguments:
        - inputs (list): input time-series data
        - outputs (list): output time-series data
        - nf (float): the number of steps the encoder must calculate from the past
        - T-truncation time (float): the number of steps the encoder must predict to the future
        - split_fraction (float): the ratio to split the data (for 0.2 the validation will be 20% and the training is the remaining 80%)
    Returns:
        - train_data_in1 (list): first half of the input training data
        - train_data_in2 (list): second half of the input training data
        - valid_data_in (list): input validation data between the first and second training data
        - train_data_out1 (list): first half of the output training data
        - train_data_out2 (list): second half of the output training data
        - valid_data_out (list): output validation data between the first and second training data
    """

    split_index = random.randint(nf+T, int(len(inputs)*(1-split_fraction)-(nf+T)))

    valid_data_in = inputs[split_index:split_index + int(len(inputs) * split_fraction)]
    train_data_in1 = inputs[:split_index]
    train_data_in2 = inputs[split_index + int(len(inputs) * split_fraction):]

    valid_data_out = outputs[split_index:split_index + int(len(outputs) * split_fraction)]
    train_data_out1 = outputs[:split_index]
    train_data_out2 = outputs[split_index + int(len(outputs) * split_fraction):]

    return train_data_in1, train_data_in2, valid_data_in, train_data_out1, train_data_out2, valid_data_out

def create_random_train_test_split(data_folders, n_lag, T, split_fraction = 0.2):
    """
    Creates random training and validation data from time-series data.

    Arguments:
        - data_folders (string): path direction of the time-series data file
        - nf (float): the number of steps the encoder must calculate from the past
        - T-truncation time (float): the number of steps the encoder must predict to the future
        - split_fraction (float): the ratio to split the data (for 0.2 the validation will be 20% and the training is the remaining 80%)
    Returns:
        - train_data (deepSI.system_data): training data
        - valid_data (deepSI.system_data): validation data
    """

    input_train_data = []
    input_valid_data = []
    output_train_data = []
    output_valid_data = []

    data_names_list = os.listdir(data_folders)
    for name in data_names_list:
        folder = os.path.join(data_folders, name)
        inputs = csv_read(folder, ['input.csv'])
        outputs = csv_read(folder, ['output.csv'])

        train_data_in1, train_data_in2, valid_data_in, train_data_out1, train_data_out2, valid_data_out = split_list(inputs, outputs, nf=n_lag, T=T, split_fraction=split_fraction)

        input_train_data.append(train_data_in1)
        input_train_data.append(train_data_in2)
        output_train_data.append(train_data_out1)
        output_train_data.append(train_data_out2)
        input_valid_data.append(valid_data_in)
        output_valid_data.append(valid_data_out)

    train_data_list = []
    valid_data_list = []
    for input1, output1 in zip(input_train_data, output_train_data):
            sys_data_part_train = deepSI.System_data(u=input1, y=output1)
            train_data_list.append(sys_data_part_train)

    for input2, output2 in zip(input_valid_data, output_valid_data):
            sys_data_part_valid = deepSI.System_data(u=input2, y=output2)
            valid_data_list.append(sys_data_part_valid)

    train_data = deepSI.System_data_list(sys_data_list=train_data_list)
    valid_data = deepSI.System_data_list(sys_data_list=valid_data_list)

    return train_data, valid_data


