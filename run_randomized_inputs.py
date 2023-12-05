import pyPanSim as sp
import numpy as np
import random
import os
from datetime import datetime
import csv

def get_results(resultArray):
  nHospitalied = []
  for result in resultArray:
    hosp = 0.1*result[6] + 0.1*result[7]  # hospitalized people + more severe hospitalied people
    nHospitalied.append(hosp)
  return nHospitalied

def get_rnd_runoptions(input_sets):
  rnd = random.randint(0, 17)
  return rnd, input_sets[rnd]

def write_single(results, filename):  # writer in case of Single Input or Single Output cases
  with open(filename, 'w') as myfile:
    writer = csv.writer(myfile)
    for result in results:
      writer.writerow([result])

# Datafolder structure
now = datetime.now()
strNow = now.strftime("%Y_%m_%d_%H%M%S")
strFolder = "SimData/Randomized_inputs_" + strNow
   

simulator = sp.SimulatorInterface()
init_options = ['panSim', '-r', ' ', '--diags', '0', '--quarantinePolicy', '0', '-k', '0.00041',
                '--progression', 'inputConfigFiles/progressions_Jun17_tune/transition_config.json',
                '-A', 'inputConfigFiles/agentTypes_3.json',
                '-a', 'inputRealExample/agents1.json',
                '-l', 'inputRealExample/locations0.json',
                '--infectiousnessMultiplier', '0.98,1.81,2.11,2.58,4.32,6.8,6.8',
                '--diseaseProgressionScaling', '0.94,1.03,0.813,0.72,0.57,0.463,0.45',
                '--closures','inputConfigFiles/emptyRules.json'
                ]
simulator.initSimulation(init_options)

input_sets = [["TPdef", "PLNONE", "CFNONE", "SONONE", "QU0", "MA1.0"], ["TPdef", "PL0", "CFNONE", "SONONE", "QU0", "MA1.0"], ["TPdef", "PLNONE", "CF2000-0500", "SONONE", "QU0", "MA1.0"],
              ["TPdef", "PLNONE", "CFNONE", "SO12", "QU0", "MA1.0"], ["TPdef", "PLNONE", "CFNONE", "SO3", "QU0", "MA1.0"], ["TPdef", "PLNONE", "CFNONE", "SONONE", "QU1", "MA1.0"],
              ["TPdef", "PLNONE", "CFNONE", "SONONE", "QU2", "MA1.0"], ["TPdef", "PLNONE", "CFNONE", "SONONE", "QU0", "MA0.8"], ["TP015", "PLNONE", "CFNONE", "SONONE", "QU1", "MA0.8"],
              ["TP015", "PLNONE", "CFNONE", "SONONE", "QU2", "MA1.0"], ["TP015", "PLNONE", "CFNONE", "SO12", "QU1", "MA1.0"], ["TP015", "PLNONE", "CFNONE", "SO3", "QU1", "MA1.0"],
              ["TP015", "PLNONE", "CFNONE", "SO12", "QU2", "MA1.0"], ["TP015", "PLNONE", "CFNONE", "SO3", "QU2", "MA1.0"], ["TP015", "PLNONE", "CFNONE", "SONONE", "QU1", "MA0.8"],
              ["TP035", "PLNONE", "CFNONE", "SONONE", "QU2", "MA0.8"], ["TP035", "PL0", "CFNONE", "SO3", "QU2", "MA0.8"], ["TP035", "PLNONE", "CF2000-0500", "SO3", "QU2", "MA0.8"]]

# initial run options
run_options = input_sets[0]
input_idx = 0

CONSTANT_PERIOD = 7  # run every input for 7 days
ENDTIME = 250  # days

results_agg = []
inputs_agg = []
day_counter = 0
for i in range(0, ENDTIME):
    results = simulator.runForDay(run_options)
    day_counter += 1
    if day_counter == CONSTANT_PERIOD:
       input_idx, run_options = get_rnd_runoptions(input_sets)
       day_counter = 0
    results_agg.append(results)
    inputs_agg.append(input_idx)


print("Simulation finished...")

nHospitalized = get_results(results_agg)

# Saving the results
currentDir = os.getcwd()
#path = os.path.join(currentDir, )
os.mkdir(strFolder)

strInputPath = os.path.join(strFolder, "input.csv")
strOutputPath = os.path.join(strFolder, "output.csv")

write_single(inputs_agg, strInputPath)
write_single(nHospitalized, strOutputPath)

print(f"Results saved into {strFolder} directory.")
