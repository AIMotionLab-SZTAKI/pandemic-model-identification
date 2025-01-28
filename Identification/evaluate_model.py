import deepSI
import os
from utils import *

encoder_id = "2024_11_20_134003_OPnw3Q"
nu = 1

#encoder_id = "2024_03_18_121141_eAO3Gw"
#nu = 6

# Load test date
TestDataFolderName = "SimData/Test"
CurrDir = os.getcwd()
test_data_folders = os.path.join(CurrDir, TestDataFolderName)
testdata = create_sysdata_from_file(test_data_folders, nu)

# Load the encoder:
fitsys = deepSI.load_system(f"EncoderData/{encoder_id}/best.pt")
test_results = fitsys.apply_experiment(testdata, save_state=True)

RMS_test = test_results.RMS(testdata)
NRMS_test = test_results.NRMS(testdata)
print(f'Test RMS error: {RMS_test}')
print(f'Test NRMS error: {NRMS_test:.2%}')

if nu == 1:
    plot_SISO_results(test_results, testdata, blockfig=True)
else:
    plot_MISO(test_results, testdata, blockfig=True)
