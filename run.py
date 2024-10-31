import datetime
import os
import subprocess
import shutil
import json
from params import *
from data_generation import *


print("Writing Info File...")
dir_name = "Runs/" + datetime.datetime.now().strftime("run %d.%m.%y - %H.%M.%S") + "/"

os.mkdir(dir_name)
os.mkdir(dir_name + input_dir)
os.mkdir(dir_name + output_dir)
os.mkdir(dir_name + val_dir)
info = {
    "STEP_T": STEP_T,
    "STEP_X": STEP_X,
    "END_T": END_T,
    "END_X": END_X,
    "FUNC_GENERATOR": FUNC_GENERATOR.__name__,
    "EXPERIMENT_INFO": EXPERIMENT_INFO
}

infofile = open(dir_name + infofile_name, "w")
json.dump(info, infofile, indent=4)
infofile.close()

print("Generating functions for validation...")
if IS_SINGLE_RUN:
    generate_single(STEP_T, STEP_X, END_T, END_X, FUNC_GENERATOR, 
                    dir_name + input_dir + datafile_name + datafile_format,
                    dir_name + val_dir + datafile_name + datafile_format)

def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True, shell=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)

print("Running model...")
params = ["dotnet", 
          "run", 
          "--project", 
          "InverseProblemBN.csproj", 
          str(STEP_T), str(STEP_X), str(END_T), str(END_X), 
          dir_name + input_dir + datafile_name + datafile_format,
          dir_name + output_dir + datafile_name + datafile_format]
for path in execute(params):
    print(path, end="")

print("Running notebook...")
shutil.copy("Notebooks/single_run.ipynb", dir_name + "single_run.ipynb")
params = "jupyter nbconvert --execute --to notebook --inplace \"" + dir_name + "single_run.ipynb\""
for path in execute(params):
    print(path, end="")

