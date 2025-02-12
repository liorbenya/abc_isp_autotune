import sys, getopt
import configparser
from deap.benchmarks import *
import ast
import os
import shutil
from dataclasses import dataclass
import csv
from image_eval import calculate_brisque
from from_paramters_to_json import update_json_fiels_for_rsim
sys.path.append("C:/Users/liorby/Documents/src/cv-pipelines/Hailo_mercury_Rsimu_case/")
from  RunSimulator import RunSimulator
from threading import Lock

last_result = 0
csv_file_path = "C:/Users/liorby/Downloads/ABCPython-master/report.csv"
result_path = "C:/Users/liorby/Downloads/ABCPython-master/results"
lock_csv = Lock()
lock_last_result = Lock()

def get_json_file_path(individual):
    output_file_path = "results/res_config"
    # res_file_path = output_file_path + "_factor_" + str(individual[0]) + "_sigma_" + ("%.1f" % individual[1]) + ".json"
    res_file_path = output_file_path
    for param in individual:
        res_file_path += "_" + str(param)
    res_file_path += ".json"
    return res_file_path

def add_info_to_csv(score, params):
    # new_line = [str(params[0]), params[1], score]
    new_line = [str(param) for param in params] + [score]
    lock_csv.acquire()
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Append a single row
        writer.writerow(new_line)
    lock_csv.release()

@dataclass
class ParamsOptions:
    minBound: float
    maxBound: float
    jump: float
    type: str

defualt_params = { 
    "dmsc_sharpen_factor" :  ParamsOptions(minBound=0, maxBound=511, jump=50, type="int") ,
                   "sigma" : ParamsOptions(minBound=0.1, maxBound=16.0, jump=0.5, type="float") 
}

defualt_params_V2 = { 
    "dmsc_sharpen_factor_white" :  ParamsOptions(minBound=0, maxBound=511, jump=50, type="int") ,
    "dmsc_sharpen_factor_black" :  ParamsOptions(minBound=0, maxBound=511, jump=50, type="int") ,
    "dmsc_sharpen_clip_white" :  ParamsOptions(minBound=0, maxBound=2047, jump=50, type="int") ,
    "dmsc_sharpen_clip_black" :  ParamsOptions(minBound=0, maxBound=2047, jump=50, type="int") ,
                   "sigma" : ParamsOptions(minBound=0.1, maxBound=16.0, jump=0.5, type="float"),
                   "strength" : ParamsOptions(minBound=1, maxBound=128, jump=10, type="int"),

}

defualt_params_V3 = {
        "dmsc_sharpen_factor_white" :  ParamsOptions(minBound=0, maxBound=511, jump=50, type="int") ,
        "dmsc_sharpen_factor_black" :  ParamsOptions(minBound=0, maxBound=511, jump=50, type="int") ,
        "dmsc_sharpen_clip_white" :  ParamsOptions(minBound=0, maxBound=2047, jump=50, type="int") ,
        "dmsc_sharpen_clip_black" :  ParamsOptions(minBound=0, maxBound=2047, jump=50, type="int") ,
        "dmsc_sharpen_size": ParamsOptions(minBound=0, maxBound=16, jump=2, type="int"),
        "dmsc_sharpen_t1": ParamsOptions(minBound=0, maxBound=2047, jump=50, type="int"),
        "dmsc_sharpen_t2_shift": ParamsOptions(minBound=0, maxBound=11, jump=1, type="int"),
        "dmsc_sharpen_t3": ParamsOptions(minBound=0, maxBound=2047, jump=50, type="int"),
        "dmsc_sharpen_t4_shift": ParamsOptions(minBound=0, maxBound=11, jump=1, type="int"),
                   "sigma" : ParamsOptions(minBound=0.1, maxBound=16.0, jump=0.5, type="float"),
                   "strength" : ParamsOptions(minBound=1, maxBound=128, jump=10, type="int"),
}

def constraint(params, params_key):
    dict =zip(params_key, params)
    if ("dmsc_sharpen_t1" in params_key and
         "dmsc_sharpen_t2_shift" in params_key and 
         "dmsc_sharpen_t3" in params_key):
        res =  dict["dmsc_sharpen_t1"] + 2**(dict["dmsc_sharpen_t2_shift"]) < dict["dmsc_sharpen_t3"]
        if (not res):
            return [False, ["dmsc_sharpen_t1", "dmsc_sharpen_t2_shift" , "dmsc_sharpen_t3"]]
    return [True, []]

ITERATOIN_NUM = 100

class Config:
    def __init__(_self, argv):
            config = configparser.ConfigParser()
            config.read(os.path.dirname(os.path.abspath(__file__))+'/ABC.ini')
            #####SETTINGS FILE######
            _self.OBJECTIVE_FUNCTION = _self.objFunctionSelector.get(config['DEFAULT']['ObjectiveFunction'], "Error")
            _self.NUMBER_OF_POPULATION = int(config['DEFAULT']['NumberOfPopulation'])
            _self.MAXIMUM_EVALUATION = int(config['DEFAULT']['MaximumEvaluation'])
            _self.LIMIT = int(config['DEFAULT']['Limit'])
            _self.FOOD_NUMBER = int(_self.NUMBER_OF_POPULATION / 2)
            _self.DIMENSION = int(config['DEFAULT']['Dimension'])
            # _self.PARAMETERS_LIST = config['DEFAULT']['Paramters_List'].split(", ")
            # import ipdb; ipdb.set_trace()
            _self.PARAMETERS_DICT = defualt_params_V2
            _self.PARAMETERS_LIST = list(_self.PARAMETERS_DICT.keys())
            _self.DIMENSION = len(_self.PARAMETERS_LIST)
            _self.UPPER_BOUND = float(config['DEFAULT']['UpperBound'])
            _self.LOWER_BOUND = float(config['DEFAULT']['LowerBound'])
            _self.MINIMIZE = bool(config['DEFAULT']['Minimize'] == 'True')
            _self.RUN_TIME = int(config['DEFAULT']['RunTime'])
            _self.SHOW_PROGRESS = bool(config['REPORT']['ShowProgress']=='True')
            _self.PRINT_PARAMETERS = bool(config['REPORT']['PrintParameters']=='True')
            _self.RUN_INFO = bool(config['REPORT']['RunInfo']=='True')
            _self.RUN_INFO_COMMANDLINE = bool(config['REPORT']['CommandLine']=='True')
            _self.SAVE_RESULTS = bool(config['REPORT']['SaveResults']=='True')
            _self.RESULT_REPORT_FILE_NAME = config['REPORT']['ResultReportFileName']
            _self.PARAMETER_REPORT_FILE_NAME = config['REPORT']['ParameterReportFileName']
            _self.RESULT_BY_CYCLE_FOLDER = config['REPORT']['ResultByCycleFolder']
            _self.OUTPUTS_FOLDER_NAME = str(config['REPORT']['OutputsFolderName'])
            _self.RANDOM_SEED = config['SEED']['RandomSeed'] == 'True'
            _self.SEED = int(config['SEED']['Seed'])
            _self.PARAMS_POSSIBLE_VALUES = _self.create_params_possiable_values()
            _self.last_result = 0
            _self.ietr_left = ITERATOIN_NUM
            _self.WORKERS = int(config['DEFAULT']['NUM_WORKERS'])
            _self.CONSTRAINT_FUNC = constraint
            data_row =  _self.PARAMETERS_LIST + ['score']
            with open(csv_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                # Write data into the CSV file
                writer.writerows(data_row)

            # Delete the entire folder and its contents
            shutil.rmtree(result_path)
            os.makedirs(result_path)

            #####SETTINGS FILE######

            #####SETTINGS ARGUMENTS######
            try:
                opts, args = getopt.getopt(argv, 'hn:m:t:d:l:u:r:o:',
                                           ['help', 'np=', 'max_eval=','trial=', 'dim=', 'lower_bound=', 'upper_bound=', 'runtime=',
                                            'obj_fun=','output_folder=','file_name=','param_name=','res_cycle_folder=','show_functions'])
            except getopt.GetoptError:
                print('Usage: ABCAlgorithm.py -h or --help')
                sys.exit(2)
            for opt, arg in opts:
                if opt in ('-h', '--help'):
                    print('-h or --help : Show Usage')
                    print('-n or --np : Number of Population')
                    print('-m or --max_eval : Maximum Evaluation')
                    #print('-t or --trial : Maximum Trial')
                    print('-d or --dim : Dimension')
                    print('-l or --lower_bound : Lower Bound')
                    print('-u or --upper_bound : Upper Bound')
                    print('-r or --runtime : Run Time')
                    print('-o or --obj_fun : Objective Function')
                    print('--show_functions : Show Objective Functions')
                    print('--output_folder= [DEFAULT: Outputs]')
                    print('--file_name= [DEFAULT: Run_Results.csv]')
                    print('--param_name= [DEFAULT: Param_Results.csv]')
                    print('--res_cycle_folder= [DEFAULT: ResultByCycle]')

                    sys.exit()
                elif opt in ('-n', '--np'):
                    _self.NUMBER_OF_POPULATION = int(arg)
                elif opt in ('-m', '--max_eval'):
                    _self.MAXIMUM_EVALUATION = int(arg)
                elif opt in ('-d', '--dim'):
                    _self.DIMENSION = int(arg)
                elif opt in ('-t', '--trial'):
                    _self.LIMIT = int(arg)
                elif opt in ('-l', '--lower_bound'):
                    _self.LOWER_BOUND = float(arg)
                elif opt in ('-u', '--upper_bound'):
                    _self.UPPER_BOUND = float(arg)
                elif opt in ('-r', '--runtime'):
                    _self.RUN_TIME = int(arg)
                elif opt in ('-o', '--obj_fun'):
                    _self.OBJECTIVE_FUNCTION = _self.objFunctionSelector.get(arg, "sphere")
                elif opt in ('--output_folder'):
                    _self.OUTPUTS_FOLDER_NAME = arg
                elif opt in ('--param_name'):
                    _self.PARAMETER_REPORT_FILE_NAME = arg
                elif opt in ('--file_name'):
                    _self.RESULT_REPORT_FILE_NAME = arg
                elif opt in ('--res_cycle_folder'):
                    _self.RESULT_BY_CYCLE_FOLDER = arg
                elif opt in ('--show_functions'):
                    print("We use deap.benchmarks functions. Available functions are listed below:")
                    for i in _self.objFunctionSelector:
                        print(i)
                    sys.exit()
            #####SETTINGS ARGUMENTS######

    def create_params_possiable_values(_self):
        params_possiable_vals = {}
        for key, value in _self.PARAMETERS_DICT.items():
            new_lst = []
            val = value.minBound
            while val <= value.maxBound:
                new_lst.append(val)
                val += value.jump
            if (val != value.maxBound):
                new_lst.append(value.maxBound)
            params_possiable_vals[key] = new_lst
        return params_possiable_vals

    def user_defined_function(individual):
        global last_result
        json_path = "C:/Users/liorby/Downloads/ABCPython-master/sfr_plus_cfg.json"
        res_file_path = get_json_file_path(individual)
        try :
            config_name = res_file_path.split('/')[-1]
            #Save the image
            avg_image_name = config_name.replace('.json', '.png')
            img_path = os.path.join(result_path, avg_image_name)
            score = calculate_brisque(img_path)
            add_info_to_csv(score, individual)
            return [score, False]
        except:
            pass
        lock_last_result.acquire()
        interm_res_dir = result_path+ "/" + "result" + str(last_result)
        # avg_image_dir = result_path + "/" + "avg_images"  + str(last_result)
        last_result+=1
        # update_json_fiels_for_rsim(json_path, individual[0], individual[1], res_file_path, interm_res_dir+ "/")
        update_json_fiels_for_rsim(json_path, individual, list(defualt_params_V2.keys()), res_file_path, interm_res_dir+ "/")
        lock_last_result.release()
        # import ipdb; ipdb.set_trace()
        img_path = RunSimulator(result_path, res_file_path, interm_res_dir, result_path)
        score = calculate_brisque(img_path)
        add_info_to_csv(score, individual)
        return [score, True]

    #######FUNCTION_LIST######
    objFunctionSelector = {
        'sphere': sphere,
        'rastrigin': rastrigin,
        'rosenbrock': rosenbrock,
        'rand': rand,
        'plane': plane,
        'cigar': cigar,
        'h1': h1,
        'ackley': ackley,
        'bohachevsky': bohachevsky,
        'griewank': griewank,
        'rastrigin_scaled': rastrigin_scaled,
        'rastrigin_skew': rastrigin_skew,
        'schaffer': schaffer,
        'schwefel': schwefel,
        'himmelblau': himmelblau,
        'user_defined': user_defined_function
    }
    #######FUNCTION_LIST######
