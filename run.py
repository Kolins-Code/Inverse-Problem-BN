import datetime
import os
import subprocess
import shutil
import json
from params import *
from data_generation import *
from tiknonov_regularization import *


print("Writing Info File...")
dir_name = "Runs/" + datetime.datetime.now().strftime("run %d.%m.%y - %H.%M.%S") + "/"

os.mkdir(dir_name)
os.mkdir(dir_name + input_dir)
os.mkdir(dir_name + output_dir)
os.mkdir(dir_name + val_dir)
os.mkdir(dir_name + reg_dir)
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
    if IS_LAPLACE == 1:
        generate_laplace(STEP_X / 2, END_X, 2, lambda x: np.zeros(x.shape), FUNC_GENERATOR, 
                        dir_name + input_dir + "derivative_" + datafile_name + datafile_format,
                        dir_name + input_dir + datafile_name + datafile_format,
                        dir_name + val_dir + datafile_name + datafile_format)
    elif IS_LAPLACE == 2 or IS_LAPLACE == 3:
        generate_laplace_coef(STEP_X / 2, END_X, 2, FUNC_GENERATOR,
                        dir_name + input_dir + "1" + datafile_name + datafile_format,
                        dir_name + input_dir + "2" + datafile_name + datafile_format,
                        dir_name + input_dir + "1derivative_" + datafile_name + datafile_format,
                        dir_name + input_dir + "2derivative_" + datafile_name + datafile_format,
                        dir_name + val_dir + datafile_name + datafile_format)
    else:
        generate_single(STEP_T / 10, STEP_X, END_T, END_X, FUNC_GENERATOR, 
                    dir_name + input_dir + datafile_name + datafile_format,
                    dir_name + val_dir + datafile_name + datafile_format)
    #generate_harmonic(STEP_T, STEP_X, END_T, END_X, FUNC_GENERATOR, 
    #                  dir_name + input_dir + datafile_name + datafile_format,
    #                  dir_name + val_dir + datafile_name + datafile_format,
    #                  10)

if RUN_REGULARIZATION:
    print("Running regularization model...")
    solver = RegularizationSolver(STEP_T, STEP_X, END_T, END_X, ERROR_REG,
                                dir_name + input_dir + datafile_name + datafile_format,
                                dir_name + reg_dir + datafile_name + datafile_format)
    solver.solve()

def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True, shell=True, encoding='utf-8')
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)

print("Running model...")
if IS_LAPLACE == 1:
    params = ["dotnet", 
            "run", 
            "--project", 
            "InverseProblemBNLaplace.csproj", 
            str(STEP_X), str(END_X), 
            dir_name + input_dir + datafile_name + datafile_format,
            dir_name + input_dir + "derivative_" + datafile_name + datafile_format,
            dir_name + output_dir + datafile_name + datafile_format]
elif IS_LAPLACE == 2:
    params = ["dotnet", 
            "run", 
            "--project", 
            "InverseProblemBNLaplace2.csproj", 
            str(STEP_X), str(END_X), 
            dir_name + input_dir + "1" + datafile_name + datafile_format,
            dir_name + input_dir + "2" + datafile_name + datafile_format,
            dir_name + input_dir + "1derivative_" + datafile_name + datafile_format,
            dir_name + input_dir + "2derivative_" + datafile_name + datafile_format,
            dir_name + output_dir + datafile_name + datafile_format]
elif IS_LAPLACE == 3:
    params = ["dotnet", 
            "run", 
            "--project", 
            "InverseProblemBNLaplace3.csproj", 
            str(STEP_X), str(END_X), 
            dir_name + input_dir + "1" + datafile_name + datafile_format,
            dir_name + input_dir + "2" + datafile_name + datafile_format,
            dir_name + input_dir + "1derivative_" + datafile_name + datafile_format,
            dir_name + input_dir + "2derivative_" + datafile_name + datafile_format,
            dir_name + output_dir + datafile_name + datafile_format]
else:
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

