import deepSI
import os
import numpy as np
from datetime import datetime
import pickle

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
