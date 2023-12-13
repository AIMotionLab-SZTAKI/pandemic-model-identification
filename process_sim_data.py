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
    idxMax = output.argmax()
    inputMulti = []
    for data in input:
        inputMulti.append(get_input_codes(data))
    if int(1.5*idxMax) < len(output):
        out_trimmed = output[:int(1.25*idxMax)]
        input_trimmed = input[:int(1.25*idxMax)]
        inputMulti_trimmed = inputMulti[:int(1.25*idxMax)]
    else:
        out_trimmed = output
        input_trimmed = input
        inputMulti_trimmed = inputMulti
    csv_write(inputMulti, folderPath, ["inputMulti.csv"], timeseries=True)
    csv_write(out_trimmed, folderPath, ["outputTrimmed.csv"])
    csv_write(input_trimmed, folderPath, ["inputTrimmed.csv"])
    csv_write(inputMulti_trimmed, folderPath, ["inputMultiTrimmed.csv"])
print("Training data processed.")

for folder in valid_folder_list:
    folderPath = os.path.join(valid_folder, folder)
    input = csv_read(folderPath, ['input.csv'])
    output = csv_read(folderPath, ['output.csv'])
    idxMax = output.argmax()
    inputMulti = []
    for data in input:
        inputMulti.append(get_input_codes(data))
    if int(1.5*idxMax) < len(output):
        out_trimmed = output[:int(1.25*idxMax)]
        input_trimmed = input[:int(1.25*idxMax)]
        inputMulti_trimmed = inputMulti[:int(1.25*idxMax)]
    else:
        out_trimmed = output
        input_trimmed = input
        inputMulti_trimmed = inputMulti
    csv_write(inputMulti, folderPath, ["inputMulti.csv"], timeseries=True)
    csv_write(out_trimmed, folderPath, ["outputTrimmed.csv"])
    csv_write(input_trimmed, folderPath, ["inputTrimmed.csv"])
    csv_write(inputMulti_trimmed, folderPath, ["inputMultiTrimmed.csv"])
print("Validation data processed.")

for folder in test_folder_list:
    folderPath = os.path.join(test_folder, folder)
    input = csv_read(folderPath, ['input.csv'])
    output = csv_read(folderPath, ['output.csv'])
    idxMax = output.argmax()
    inputMulti = []
    for data in input:
        inputMulti.append(get_input_codes(data))
    if int(1.5*idxMax) < len(output):
        out_trimmed = output[:int(1.25*idxMax)]
        input_trimmed = input[:int(1.25*idxMax)]
        inputMulti_trimmed = inputMulti[:int(1.25*idxMax)]
    else:
        out_trimmed = output
        input_trimmed = input
        inputMulti_trimmed = inputMulti
    csv_write(inputMulti, folderPath, ["inputMulti.csv"], timeseries=True)
    csv_write(out_trimmed, folderPath, ["outputTrimmed.csv"])
    csv_write(input_trimmed, folderPath, ["inputTrimmed.csv"])
    csv_write(inputMulti_trimmed, folderPath, ["inputMultiTrimmed.csv"])
print("Test data processed.")

