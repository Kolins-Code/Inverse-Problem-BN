from start_func_generation import *


STEP_X = 0.01
STEP_T = 0.0005 #STEP_X ** 2 / 2


END_X = 1
END_T = 0.01 #STEP_T * 50

FUNC_GENERATOR = normal_dist_func

IS_SINGLE_RUN = True

EXPERIMENT_INFO = "Модель с VMP на основе условий на сетку через ConstrainEqualRandom. Тест на 20 слоях с dt=0.005"


datafile_name = "data"
datafile_format = ".csv"
infofile_name = "info.json"
input_dir = "input/"
output_dir = "output/"
val_dir = "val/"
