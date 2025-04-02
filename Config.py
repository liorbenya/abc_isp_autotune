import sys, getopt
import configparser
from deap.benchmarks import *
import ast
import os
import shutil
from dataclasses import dataclass
import csv
import random
import numpy as np
from image_eval import calculate_brisque
from from_paramters_to_json import update_json_fiels_for_rsim
sys.path.append("C:/Users/hailo/src/cv-pipelines/Hailo_mercury_Rsimu_case/")
from  RunSimulator import RunSimulator
# sys.path.append("C:/Users/hailo/src/cv-pipelines/Utils/")
# from image_utils import 
import multiprocessing



last_result = 0
csv_file_path = "C:/Users/hailo/src/abc_isp_autotune/report.csv"
threads_csv = "C:/Users/hailo/src/abc_isp_autotune/threads_report.txt"
result_path = "C:/Users/hailo/src/abc_isp_autotune/results"
lock_csv = multiprocessing.Lock()
lock_last_result = multiprocessing.Lock()
lock_threads = multiprocessing.Lock()

lock_csv_2 = multiprocessing.Lock()
lock_threads_2 = multiprocessing.Lock()

def score_function(individual):
    global last_result
    json_path = "C:/Users/hailo/src/abc_isp_autotune/sfr_plus_cfg.json"
    raw_dir =  "C:/Users/hailo/src/abc_isp_autotune/original_raws/"
    raw_files = [f for f in os.listdir(raw_dir) 
                if f.lower().endswith('.raw')]
    raw_files = ["C:/Users/hailo/src/abc_isp_autotune/original_raws/lab15_3ms_gain1_imx678_white_women.raw"]
    score = 0  
    for raw_file in raw_files:
        lock_last_result.acquire()
        raw_path = os.path.join(raw_dir, raw_file)
        res_file_path = get_json_file_path(individual)
        rand =  random.randint(1, 512)
        interm_res_dir = result_path+ "/" + "result" + str(last_result) + '_' + str(rand)
        # avg_image_dir = result_path + "/" + "avg_images"  + str(last_result)
        last_result+=1
        new_raw_path = copy_raws(raw_path)
        # update_json_fiels_for_rsim(json_path, individual[0], individual[1], res_file_path, interm_res_dir+ "/")
        add_thrads_info()
        update_json_fiels_for_rsim(json_path, individual, get_params_dict_fields(defualt_params_V4), res_file_path, interm_res_dir+ "/", new_raw_path, defualt_params_V4.keys())
        lock_last_result.release()
        try:
            img_path = RunSimulator(result_path, res_file_path, interm_res_dir, result_path)
            score += calculate_brisque(img_path)
        except ValueError as err:
            print(err)
            return 100, False
        os.remove(new_raw_path)
        shutil.rmtree(interm_res_dir)
    score = score/len(raw_files)
    add_info_to_csv(score, individual)
    return [score, True]

def copy_raws(raw_path):
    rand = random.randint(1, 500)
    output_path = "C:/Users/hailo/src/abc_isp_autotune/raws/image_" + str(rand) +".raw"
    # Copy the .raw file
    shutil.copy(raw_path, output_path)
    return output_path

def get_json_file_path(individual):
    output_file_path = "results/res_"
    # res_file_path = output_file_path + "_factor_" + str(individual[0]) + "_sigma_" + ("%.1f" % individual[1]) + ".json"
    res_file_path = output_file_path + str(last_result) +"_config"
    for param in individual:
        res_file_path += "_" + str(param)
    res_file_path += ".json"
    return res_file_path

def add_thrads_info():
    lock_threads.acquire()
    with open(threads_csv, mode = "a", newline='') as tfile:
        with lock_threads_2:
            line = "thread: " + str(multiprocessing.current_process()) + " got: " + str(last_result) + "\n"
            tfile.write(line)
            tfile.flush()
    lock_threads.release()

def add_info_to_csv(score, params):
    # new_line = [str(params[0]), params[1], score]
    new_line = [str(param) for param in params] + [score]
    lock_csv.acquire()
    with open(csv_file_path, mode='a', newline='') as file:
        with lock_csv_2:
            writer = csv.writer(file)
            # Append a single row
            writer.writerow(new_line)
            file.flush()
    lock_csv.release()

@dataclass
class ParamsOptions:
    minBound: float
    maxBound: float
    jump: float
    type: str

#Params to work on:
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
possiable_classes = ["CDmscv2", "C2dnrv3", "CWdrv41", "CCproc", "CEEv1", "CCpdv1"]

defualt_params_V4 = {
    "CDmscv2" : {
        "dmsc_sharpen_factor_white" :  ParamsOptions(minBound=0, maxBound=511, jump=50, type="int") ,
        "dmsc_sharpen_factor_black" :  ParamsOptions(minBound=0, maxBound=511, jump=50, type="int") ,
        "dmsc_sharpen_clip_white" :  ParamsOptions(minBound=0, maxBound=2047, jump=50, type="int") ,
        "dmsc_sharpen_clip_black" :  ParamsOptions(minBound=0, maxBound=2047, jump=50, type="int") ,
        "dmsc_sharpen_size": ParamsOptions(minBound=0, maxBound=16, jump=2, type="int"),
        "dmsc_sharpen_t1": ParamsOptions(minBound=0, maxBound=2047, jump=50, type="int"),
        "dmsc_sharpen_t2_shift": ParamsOptions(minBound=0, maxBound=11, jump=1, type="int"),
        "dmsc_sharpen_t3": ParamsOptions(minBound=0, maxBound=2047, jump=50, type="int"),
        "dmsc_sharpen_t4_shift": ParamsOptions(minBound=0, maxBound=11, jump=1, type="int")
    },
    "C2dnrv3" : { "sigma" : ParamsOptions(minBound=0.1, maxBound=16.0, jump=0.5, type="float"),
                   "strength" : ParamsOptions(minBound=1, maxBound=128, jump=10, type="int")
                },
    "CWdrv41" : {
        "strength" :  ParamsOptions(minBound=1, maxBound=128, jump=10, type="int"),
        "high_strength" : ParamsOptions(minBound=1, maxBound=128, jump=10, type="int"),
        "low_strength" : ParamsOptions(minBound=0, maxBound=256, jump=10, type="int"),
        "global_strength" : ParamsOptions(minBound=1, maxBound=128, jump=10, type="int"),
        "contrast" : ParamsOptions(minBound=-1023, maxBound=1023, jump=50, type="int"),
    },
    "CCproc" : {
        "contrast" : ParamsOptions(minBound=0.3, maxBound=1.99, jump=0.1, type="float"),
        "bright" : ParamsOptions(minBound=-128, maxBound=127, jump=50, type="int"),
        "saturation" : ParamsOptions(minBound=0.0, maxBound=1.99, jump=0.1, type="float"),
    },
    "CEEv1" : {
        "ee_strength" : ParamsOptions(minBound=0, maxBound=128, jump=10, type="int"),
        "ee_y_up_gain" : ParamsOptions(minBound=0, maxBound=65535, jump=1500, type="int"),
        "ee_y_down_gain" : ParamsOptions(minBound=0, maxBound=65535, jump=1500, type="int"),
        "ee_uv_gain" : ParamsOptions(minBound=0, maxBound=65535, jump=1500, type="int"),
        "ee_edge_gain" : ParamsOptions(minBound=0, maxBound=65535, jump=1500, type="int")
    }
}


defualt_params_V5 = {
    "CDmscv2" : {
        "dmsc_sharpen_factor_white" :  ParamsOptions(minBound=0, maxBound=511, jump=50, type="int") ,
        "dmsc_sharpen_factor_black" :  ParamsOptions(minBound=0, maxBound=511, jump=50, type="int") ,
        "dmsc_sharpen_clip_white" :  ParamsOptions(minBound=0, maxBound=2047, jump=50, type="int") ,
        "dmsc_sharpen_clip_black" :  ParamsOptions(minBound=0, maxBound=2047, jump=50, type="int") ,
        "dmsc_sharpen_size": ParamsOptions(minBound=0, maxBound=16, jump=2, type="int"),
        "dmsc_sharpen_t1": ParamsOptions(minBound=0, maxBound=2047, jump=50, type="int"),
        "dmsc_sharpen_t2_shift": ParamsOptions(minBound=0, maxBound=11, jump=1, type="int"),
        "dmsc_sharpen_t3": ParamsOptions(minBound=0, maxBound=2047, jump=50, type="int"),
        "dmsc_sharpen_t4_shift": ParamsOptions(minBound=0, maxBound=11, jump=1, type="int")
    },
    "C2dnrv3" : { "sigma" : ParamsOptions(minBound=0.1, maxBound=16.0, jump=0.5, type="float"),
                   "strength" : ParamsOptions(minBound=1, maxBound=128, jump=10, type="int")
                },
    "CWdrv41" : {
        "strength" :  ParamsOptions(minBound=1, maxBound=128, jump=10, type="int"),
        "high_strength" : ParamsOptions(minBound=1, maxBound=128, jump=10, type="int"),
        "low_strength" : ParamsOptions(minBound=0, maxBound=256, jump=10, type="int"),
        "global_strength" : ParamsOptions(minBound=1, maxBound=128, jump=10, type="int"),
        "contrast" : ParamsOptions(minBound=-1023, maxBound=1023, jump=50, type="int"),
    },
    "CCproc" : {
        "contrast" : ParamsOptions(minBound=0.3, maxBound=1.99, jump=0.1, type="float"),
        "bright" : ParamsOptions(minBound=-128, maxBound=127, jump=50, type="int"),
        "saturation" : ParamsOptions(minBound=0.0, maxBound=1.99, jump=0.1, type="float"),
    },
    "CEEv1" : {
        "ee_strength" : ParamsOptions(minBound=0, maxBound=128, jump=10, type="int"),
        "ee_y_up_gain" : ParamsOptions(minBound=0, maxBound=65535, jump=1500, type="int"),
        "ee_y_down_gain" : ParamsOptions(minBound=0, maxBound=65535, jump=1500, type="int"),
        "ee_uv_gain" : ParamsOptions(minBound=0, maxBound=65535, jump=1500, type="int"),
        "ee_edge_gain" : ParamsOptions(minBound=0, maxBound=65535, jump=1500, type="int")
    },
    "CCpdv1" : {
        "bls" : ParamsOptions(minBound=0, maxBound=255*256, jump=200*256, type="int"),
    }
}

def get_params_dict_len(dict):
    len = 0 
    for class_name, class_params in dict.items():
        len += len(class_params.keys())
    return len

def get_global_params_name(class_name, field_name):
    return class_name + "/" + field_name

def get_params_dict_fields(dict):
    list_of_fields = [] 
    for class_name, class_params in dict.items():
        for key in class_params.keys():
            list_of_fields.append(get_global_params_name(class_name,key))
    return list_of_fields


def constraint(params, params_key):
    # import ipdb; ipdb.set_trace()
    zippy = zip(params_key, params)
    dictty = dict(zippy)
    if ("CDmscv2/dmsc_sharpen_t1" in params_key and
         "CDmscv2/dmsc_sharpen_t2_shift" in params_key and 
         "CDmscv2/dmsc_sharpen_t3" in params_key):
        res =  dictty["CDmscv2/dmsc_sharpen_t1"] + 2**(dictty["CDmscv2/dmsc_sharpen_t2_shift"]) < dictty["CDmscv2/dmsc_sharpen_t3"]
        if (not res):
            return [False, ["CDmscv2/dmsc_sharpen_t1", "CDmscv2/dmsc_sharpen_t2_shift" , "CDmscv2/dmsc_sharpen_t3"]]
    return [True, []]

ITERATOIN_NUM = 100

class Config:
    def __init__(_self, argv):
            config = configparser.ConfigParser()
            config.read(os.path.dirname(os.path.abspath(__file__))+'/ABC.ini')
            #####SETTINGS FILE######
            _self.NUMBER_OF_POPULATION = int(config['DEFAULT']['NumberOfPopulation'])
            _self.MAXIMUM_EVALUATION = int(config['DEFAULT']['MaximumEvaluation'])
            _self.LIMIT = int(config['DEFAULT']['Limit'])
            _self.FOOD_NUMBER = int(_self.NUMBER_OF_POPULATION/2)
            _self.DIMENSION = int(config['DEFAULT']['Dimension'])
            # import ipdb; ipdb.set_trace()
            _self.PARAMETERS_DICT = defualt_params_V4
            _self.PARAMETERS_LIST = get_params_dict_fields(_self.PARAMETERS_DICT)
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
            with open(threads_csv, mode= 'w', newline='') as tfile:
                writer = csv.writer(tfile)
                writer.writerow("")
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
                    sys.exit()
            #####SETTINGS ARGUMENTS######

    def create_params_possiable_values(_self):
        params_possiable_vals = {}
        for class_name, class_params in _self.PARAMETERS_DICT.items():
            for field_name, field_param in class_params.items():
                new_lst = []
                val = field_param.minBound
                while val <= field_param.maxBound:
                    new_lst.append(val)
                    val += field_param.jump
                if (val != field_param.maxBound):
                    new_lst.append(field_param.maxBound)
                params_possiable_vals[get_global_params_name(class_name,field_name)] = new_lst
        return params_possiable_vals
