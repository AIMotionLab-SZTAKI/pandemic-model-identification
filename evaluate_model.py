import deepSI
import os
from utils import *

encoder_id = "2023_12_06_151247_JIuHyA"

# Load test date
TestDataFolderName = "SimData/Test"
CurrDir = os.getcwd()
test_data_folders = os.path.join(CurrDir, TestDataFolderName)
testdata = create_sysdata_from_file(test_data_folders)

# Load the encoder:
fitsys = deepSI.load_system(f"EncoderData/{encoder_id}/best.pt")
test_results = fitsys.apply_experiment(testdata, save_state=True)

plot_SISO_results(test_results, testdata, blockfig=True)
