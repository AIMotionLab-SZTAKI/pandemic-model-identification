import deepSI
import os
from utils import *

encoder_id = "2023_12_13_123557_2Tf7yQ"
    #"2023_12_12_132657_oclDNA"

nu = 6

# Load test date
TestDataFolderName = "SimData/Test"
CurrDir = os.getcwd()
test_data_folders = os.path.join(CurrDir, TestDataFolderName)
testdata = create_sysdata_from_file(test_data_folders, nu, out="normal")

# Load the encoder:
fitsys = deepSI.load_system(f"EncoderData/{encoder_id}/best.pt")
test_results = fitsys.apply_experiment(testdata, save_state=True)

plot_SISO_results(test_results, testdata, blockfig=True)
