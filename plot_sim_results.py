from matplotlib import pyplot as plt
import numpy as np
import csv

def csv_read(file):
    with open(file, 'r') as myfile:
        data = np.genfromtxt(myfile, delimiter=',')
    return data

folder_name = "SimData/Moderate"

input = csv_read(folder_name + "/input.csv")
output = csv_read(folder_name + "/output.csv")

plt.figure()
plt.plot(output)
plt.xlabel("Days")
plt.ylabel("Hospitalized people")
plt.show(block=False)

plt.figure()
plt.bar(*np.unique(input, return_counts=True))
plt.xticks(np.arange(18))
plt.xlabel("Control input")
plt.ylabel("Days")
plt.show()
