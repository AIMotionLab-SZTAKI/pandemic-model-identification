# pandemic-model-identification
Identifying ANN-SS models using the SUBNET approach from data generated by a multi-agent pandemic simulator.

## Installation
First, clone the repository as
```bash
git clone https://github.com/AIMotionLab-SZTAKI/pandemic-model-identification/
```
It is advised to use a virtual environment. Create and activate it, e.g., using Linux/Bash
```bash
cd pandemic-model-identification/
python3 -m venv venv
source venv/bin/activate
```
Then, before running the identification scripts, install [deepSI](https://github.com/MaartenSchoukens/deepSI) (⚠️the implementation currently uses the legacy version⚠️):
```bash
pip install git+https://github.com/MaartenSchoukens/deepSI@legacy
```
