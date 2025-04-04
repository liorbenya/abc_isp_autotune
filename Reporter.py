import os
import shutil
import numpy as np
from decimal import Decimal

class Reporter:
    def __init__(_self, abcList):
        _self.abcList = abcList
        if(abcList[0].conf.PRINT_PARAMETERS):
            _self.print_parameters()

        if (abcList[0].conf.RUN_INFO):
            _self.run_info()
        if (abcList[0].conf.SAVE_RESULTS):
             _self.save_results()
        if (abcList[0].conf.RUN_INFO_COMMANDLINE):
            _self.command_line_print()


    def print_parameters(_self):
        for i in range(_self.abcList[0].conf.RUN_TIME):
            print(_self.abcList[i].experimentID,". run")
            for j in range(_self.abcList[0].conf.DIMENSION):
                param_name = _self.abcList[0].conf.PARAMETERS_LIST[j]
                print("Global Param[", param_name, "] ", _self.abcList[i].globalParams[j])

    def run_info(_self):
        sum = []
        best_val = 100
        best_params = [0.0] * _self.abcList[0].conf.DIMENSION
        if (_self.abcList[0].conf.MINIMIZE == False):
            best_val = 0
        for i in range(_self.abcList[0].conf.RUN_TIME):
            float_str_list = [str(num) for num in  _self.abcList[i].globalParams]
            paraams_str = "_".join(float_str_list)
            print(_self.abcList[i].experimentID + " run: ", _self.abcList[i].globalOpt, " Cycle: ",
                  i, " Params: ",paraams_str)
            if ((_self.abcList[0].conf.MINIMIZE == True and _self.abcList[i].globalOpt < best_val) or
                    (_self.abcList[0].conf.MINIMIZE == False and _self.abcList[i].globalOpt > best_val)):
                best_val = _self.abcList[i].globalOpt
                best_params = _self.abcList[i].globalParams
            sum.append(_self.abcList[i].globalOpt)
        print("Mean: ",np.mean(sum)," Std: ",np.std(sum)," Median: ",np.median(sum))
        float_str_list = [str(num) for num in best_params]
        paraams_str = "_".join(float_str_list)
        results_dir = "C:/Users/hailo/src/abc_isp_autotune/results/"
        best_res_dir =  "C:/Users/hailo/src/abc_isp_autotune/bla/"
        shutil.rmtree(best_res_dir)
        os.makedirs(best_res_dir)
        for filename in os.listdir(results_dir):
            if paraams_str in filename:
                source_path = os.path.join(results_dir, filename)
                destination_path = os.path.join(best_res_dir, filename)
                shutil.copy(source_path, destination_path)
                print(f"Copied: {filename}")



    def command_line_print(_self):
        sum = []
        for i in range(_self.abcList[0].conf.RUN_TIME):
            sum.append(_self.abcList[i].globalOpt)
        print('%1.5E' % Decimal(np.mean(sum)))

    def save_results(_self):
        if not os.path.exists(_self.abcList[0].conf.OUTPUTS_FOLDER_NAME):
            os.makedirs(_self.abcList[0].conf.OUTPUTS_FOLDER_NAME)

        header="experimentID ;Number of Population; Maximum Evaluation; Limit; Function, Dimension, Upper Bound; Lower Bound; isMinimize; Result ; Time \n"
        csvText = "{} ;{}; {}; {}; {}; {}; {}; {}; {}; {} ; {} \n"
        with open(_self.abcList[0].conf.OUTPUTS_FOLDER_NAME+"/"+_self.abcList[0].conf.RESULT_REPORT_FILE_NAME, 'a') as saveRes:
            if(sum(1 for line in open(_self.abcList[0].conf.OUTPUTS_FOLDER_NAME+"/"+_self.abcList[0].conf.RESULT_REPORT_FILE_NAME))<1):
                saveRes.write(header)

            for i in range(_self.abcList[0].conf.RUN_TIME):
                saveRes.write(csvText.format(
                    _self.abcList[i].experimentID,
                    _self.abcList[i].conf.NUMBER_OF_POPULATION,
                    _self.abcList[i].conf.MAXIMUM_EVALUATION,
                    _self.abcList[i].conf.LIMIT,
                    _self.abcList[i].conf.DIMENSION,
                    _self.abcList[i].conf.UPPER_BOUND,
                    _self.abcList[i].conf.LOWER_BOUND,
                    _self.abcList[i].conf.MINIMIZE,
                    _self.abcList[i].globalOpt,
                    _self.abcList[i].globalTime,

                ))

            header = "experimentID;"
            for j in range(_self.abcList[0].conf.DIMENSION):

                if (j < _self.abcList[0].conf.DIMENSION - 1):
                    header = header +"_"+ _self.abcList[0].conf.PARAMETERS_LIST[j] + ";"
                else:
                    header = header + "_"+ _self.abcList[0].conf.PARAMETERS_LIST[j] + "\n"
            folder_path = _self.abcList[0].conf.OUTPUTS_FOLDER_NAME + "/" + _self.abcList[0].conf.PARAMETER_REPORT_FILE_NAME

            with open(folder_path, 'w') as saveRes:
                if (sum(1 for line in open(folder_path)) < 1):
                    saveRes.write(header)

                for i in range(_self.abcList[0].conf.RUN_TIME):
                    csvText=str(_self.abcList[i].experimentID)+";"
                    for j in range(_self.abcList[0].conf.DIMENSION):
                        if(j<_self.abcList[0].conf.DIMENSION-1):
                            csvText = csvText+str(_self.abcList[i].globalParams[j])+";"
                        else:
                            csvText = csvText + str(_self.abcList[i].globalParams[j]) + "\n"
                    saveRes.write(csvText)

            for i in range(_self.abcList[0].conf.RUN_TIME):
                if not os.path.exists(_self.abcList[i].conf.OUTPUTS_FOLDER_NAME+"/"+_self.abcList[i].conf.RESULT_BY_CYCLE_FOLDER):
                    os.makedirs(_self.abcList[i].conf.OUTPUTS_FOLDER_NAME+"/"+_self.abcList[i].conf.RESULT_BY_CYCLE_FOLDER)
                with open(_self.abcList[i].conf.OUTPUTS_FOLDER_NAME+"/"+_self.abcList[i].conf.RESULT_BY_CYCLE_FOLDER+"/"+_self.abcList[i].experimentID+".txt",
                          'a') as saveRes:
                    data = "result:" 
                    for j in range(_self.abcList[i].cycle):
                        data = data + str(_self.abcList[i].globalOpts[j]) + "\n"
                        data = data + "params: " + str(_self.abcList[i].globalParams) + "\n"
                        path = "res_config"
                        for param in _self.abcList[i].globalParams:
                            path += "_" + str(param)
                        path += ".png"
                        data = data + "pic path" + path
                        saveRes.write(data)

