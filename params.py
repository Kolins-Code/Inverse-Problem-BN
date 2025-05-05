from start_func_generation import *


STEP_X = 0.01
STEP_T = 0.00005 #STEP_X ** 2 / 2


END_X = 1
END_T = 0.001 #STEP_T * 50

ERROR_REG = 1.e-4

FUNC_GENERATOR = sinh_func

IS_SINGLE_RUN = True
RUN_REGULARIZATION = True

IS_LAPLACE = 2

EXPERIMENT_INFO = "Модель с VMP на основе условий на сетку через ConstrainEqualRandom. Тест на 20 слоях с dt=0.005"


datafile_name = "data"
datafile_format = ".csv"
infofile_name = "info.json"
input_dir = "input/"
output_dir = "output/"
val_dir = "val/"
reg_dir = "reg/"
