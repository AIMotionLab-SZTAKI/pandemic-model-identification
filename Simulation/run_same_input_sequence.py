import pyPanSim as sp
import random
import os
from datetime import datetime
import csv


def get_results(resultArray):
    output = []
    for result in resultArray:
        hosp = result[6] + result[7]  # hospitalized people + more severe hospitalized people
        death = result[10]
        output.append([hosp, death])
    return output


def generate_rnd_inputs(input_sets, mode, nDays, constPeriod):
    if mode == "strict":
        lower = 13
        upper = 17
    elif mode == "loose":
        lower = 0
        upper = 4
    elif mode == "moderate":
        lower = 5
        upper = 12
    else:
        lower = 0
        upper = 17
    day_counter = 0
    input_idx = []
    input_codes = []
    rnd = random.randint(lower, upper)
    inp = input_sets[rnd]
    for i in range(nDays):
        input_idx.append(rnd)
        input_codes.append(inp)
        day_counter += 1
        if day_counter == constPeriod:
            rnd = random.randint(lower, upper)
            inp = input_sets[rnd]
            day_counter = 0
    return input_idx, input_codes

def write_aggregated(strFolder, inputs, outputs):
    file_input = os.path.join(strFolder, "aggreg_inputs.csv")
    file_output = os.path.join(strFolder, "aggreg_results.csv")
    with open(file_input, 'w') as myfile:
        writer = csv.writer(myfile)
        writer.writerows(inputs)
    with open(file_output, 'w') as myfile:
        writer = csv.writer(myfile)
        writer.writerows(outputs)

def write_multi(output, outputPath):
    with open(outputPath, 'w') as myfile:
        writer = csv.writer(myfile)
        writer.writerows(output)

def write_single(results, filename):  # writer in case of Single Input or Single Output cases
    with open(filename, 'w') as myfile:
        writer = csv.writer(myfile)
        for result in results:
            writer.writerow([result])


simulator = sp.SimulatorInterface()
init_options = ['panSim', '-r', ' ', '--diags', '0', '--quarantinePolicy', '0', '-k', '0.00041',
                '--progression', 'inputConfigFiles/progressions_Jun17_tune/transition_config.json',
                '-A', 'inputConfigFiles/agentTypes_3.json',
                '-a', 'inputRealExample/agents1.json',
                '-l', 'inputRealExample/locations0.json',
                '--infectiousnessMultiplier', '0.98,1.81,2.11,2.58,4.32,6.8,6.8',
                '--diseaseProgressionScaling', '0.94,1.03,0.813,0.72,0.57,0.463,0.45',
                '--closures', 'inputConfigFiles/emptyRules_noMutation.json'
                ]

input_sets = [["TPdef", "PLNONE", "CFNONE", "SONONE", "QU0", "MA1.0"],
              ["TPdef", "PL0", "CFNONE", "SONONE", "QU0", "MA1.0"],
              ["TPdef", "PLNONE", "CF2000-0500", "SONONE", "QU0", "MA1.0"],
              ["TPdef", "PLNONE", "CFNONE", "SO12", "QU0", "MA1.0"],
              ["TPdef", "PLNONE", "CFNONE", "SO3", "QU0", "MA1.0"],
              ["TPdef", "PLNONE", "CFNONE", "SONONE", "QU2", "MA1.0"],
              ["TPdef", "PLNONE", "CFNONE", "SONONE", "QU3", "MA1.0"],
              ["TPdef", "PLNONE", "CFNONE", "SONONE", "QU0", "MA0.8"],
              ["TP015", "PLNONE", "CFNONE", "SONONE", "QU2", "MA1.0"],
              ["TP015", "PLNONE", "CFNONE", "SONONE", "QU3", "MA1.0"],
              ["TP015", "PLNONE", "CFNONE", "SO12", "QU2", "MA1.0"],
              ["TP015", "PLNONE", "CFNONE", "SO3", "QU2", "MA1.0"],
              ["TP015", "PLNONE", "CFNONE", "SO12", "QU3", "MA1.0"],
              ["TP015", "PLNONE", "CFNONE", "SO3", "QU3", "MA1.0"],
              ["TP015", "PLNONE", "CFNONE", "SONONE", "QU2", "MA0.8"],
              ["TP035", "PLNONE", "CFNONE", "SONONE", "QU3", "MA0.8"],
              ["TP035", "PL0", "CFNONE", "SO3", "QU3", "MA0.8"],
              ["TP035", "PLNONE", "CF2000-0500", "SO3", "QU3", "MA0.8"]]

nSimulations = 5
CONSTANT_PERIOD = 7  # run every input for 7 days
ENDTIME = 250  # days
MODE = "normal"  # "normal" - randomized inputs,
#               "strict" - inputs between 13-17,
#               "loose" - inputs between 0-4
#               "moderate" - inputs between 5-12

input_idx, input_names = generate_rnd_inputs(input_sets, MODE, ENDTIME, CONSTANT_PERIOD)

for idxRun in range(nSimulations):
    # Datafolder structure
    now = datetime.now()
    strNow = now.strftime("%Y_%m_%d_%H%M%S")
    strFolder = f"RndSequence/Input_sequence_nr{idxRun}_" + strNow

    simulator.initSimulation(init_options)
    results_agg = []
    for i in range(0, ENDTIME):
        results = simulator.runForDay(input_names[i])
        results_agg.append(results)

    print(f"Simulation number {idxRun + 1} finished...")

    output = get_results(results_agg)

    # Saving the results
    os.makedirs(strFolder)

    strInputPath = os.path.join(strFolder, "input.csv")
    strOutputPath = os.path.join(strFolder, "output.csv")

    write_single(input_idx, strInputPath)
    write_multi(output, strOutputPath)

    print(f"Results saved into {strFolder} directory.")

print(f"--------------- {nSimulations} number of simulations finished ---------------")
