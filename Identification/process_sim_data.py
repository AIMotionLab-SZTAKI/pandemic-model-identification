import os
from utils import *


train_folder_list = os.listdir("SimData/Train")
valid_folder_list = os.listdir("SimData/Valid")
test_folder_list = os.listdir("SimData/Test")

currentDir = os.getcwd()
train_folder = os.path.join(currentDir, "SimData", "Train")
valid_folder = os.path.join(currentDir, "SimData", "Valid")
test_folder = os.path.join(currentDir, "SimData", "Test")

folder_list = []

for folder in train_folder_list:
    folderPath = os.path.join(train_folder, folder)
    input = csv_read(folderPath, ['input.csv'])
    output = csv_read(folderPath, ['output.csv'])
    inputMulti = []
    for data in input:
        inputMulti.append(get_input_codes(data))
    csv_write(inputMulti, folderPath, ["inputMulti.csv"], timeseries=True)
print("Training data processed.")

for folder in valid_folder_list:
    folderPath = os.path.join(valid_folder, folder)
    input = csv_read(folderPath, ['input.csv'])
    output = csv_read(folderPath, ['output.csv'])
    inputMulti = []
    for data in input:
        inputMulti.append(get_input_codes(data))
    csv_write(inputMulti, folderPath, ["inputMulti.csv"], timeseries=True)
print("Validation data processed.")

for folder in test_folder_list:
    folderPath = os.path.join(test_folder, folder)
    input = csv_read(folderPath, ['input.csv'])
    output = csv_read(folderPath, ['output.csv'])
    inputMulti = []
    for data in input:
        inputMulti.append(get_input_codes(data))
    csv_write(inputMulti, folderPath, ["inputMulti.csv"], timeseries=True)
print("Test data processed.")

