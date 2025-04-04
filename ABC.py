__author__ = "Omur Sahin"

import sys
import numpy as np
from deap.benchmarks import *
import progressbar
from Config import score_function, constraint
import multiprocessing

lock = multiprocessing.Lock()
lock_progress = multiprocessing.Lock()
lock_eval = multiprocessing.Lock()


def combine_solution(sol_idx, foods, param2change, neighbour, lst_possiable_val):
    r = random.random()
    val =  foods[sol_idx][param2change] + (
                    foods[sol_idx][param2change] - foods[neighbour][param2change]) * (
                                                            r - 0.5) * 2
    closest_value = min(lst_possiable_val, key=lambda num: abs(num - val))
    return closest_value


def get_updated_param(sol_idx, foods, param2change, neighbour, lst_possiable_val, type):
    closest_value = combine_solution(sol_idx, foods, param2change, neighbour, lst_possiable_val)
    match type:
        case "int":
            closest_value = int(closest_value)
        case "float":
            trancuate = "%.1f" % closest_value
            closest_value = float(trancuate)  
        case "double":
            trancuate = "%.1f" % closest_value
            closest_value = float(trancuate) 
    return closest_value


def update_param_multiprocess(sol_idx, foods, param2change, neighbour, lst_possiable_val, type, lst_params):
    closest_value = get_updated_param(sol_idx, foods, param2change, neighbour, lst_possiable_val, type)
    solution = np.copy(foods[sol_idx][:])
    solution[param2change] = closest_value
    res, _ = constraint(solution, lst_params)
    while (not res):
        closest_value = get_updated_param(sol_idx, foods, param2change, neighbour, lst_possiable_val, type)
        solution[param2change] = closest_value
        res, _ = constraint(solution[param2change], lst_params)
    return closest_value




def send_employed_bees_multprocess(indexes, foods, max_iterations, dimansions, food_number, lst_params, dict_possiable_val, parameters_dict, fitness, should_minimize, trial, f, prob):
    i = 0
    iter_num = 0
    
    while (i < len(indexes) and  iter_num <= max_iterations):
        index = indexes[i]
        r = random.random()
        param2change = (int)(r * dimansions)

        r = random.random()
        neighbour = (int)(r * food_number)
        while neighbour == index:
            r = random.random()
            neighbour = (int)(r * food_number)
        solution = np.copy(foods[index][:])

        r = random.random()
        param_str = lst_params[param2change]
        type =parameters_dict[param_str].type
        lst_possiable_val = dict_possiable_val[param_str]
        solution[param2change] = update_param_multiprocess(index, foods, param2change, neighbour, lst_possiable_val, type, lst_params)
        ObjValSol = score_function(solution)[0]
        FitnessSol =    np.float64(1 / (ObjValSol + 1))
        if (FitnessSol >fitness[index] and should_minimize == True) or (FitnessSol <= fitness[index] and should_minimize == False):
            trial[index] = 0
            foods[index][:] = np.copy(solution)
            f[index] = ObjValSol
            fitness[index] = FitnessSol
        else:
            trial[index] = trial[index] + 1
        i += 1
        iter_num+=1

def send_onlooker_bees_multiprocess(indexes, foods, max_iterations, dimansions, food_number, lst_params, dict_possiable_val, parameters_dict, fitness, should_minimize, trial, f, prob):
    i = 0
    iter_num = 0
    t = 0
    while (i < len(indexes) and  iter_num <= max_iterations):
        index = indexes[i]
        r = random.random()
        if ((r < prob[index] and should_minimize == True) or (r > prob[index] and should_minimize == False)):
            t+=1
            r = random.random()
            param2change = (int)(r *dimansions)
            r = random.random()
            neighbour = (int)(r * food_number)
            while neighbour == index:
                r = random.random()
                neighbour = (int)(r * food_number)
            solution = np.copy(foods[index][:])
            param_str = lst_params[param2change]
            type =parameters_dict[param_str].type
            lst_possiable_val = dict_possiable_val[param_str]
            solution[param2change] =  update_param_multiprocess(index, foods, param2change, neighbour, lst_possiable_val, type, lst_params)
            ObjValSol = score_function(solution)[0]
            FitnessSol = np.float64(1 / (ObjValSol + 1))
            if (FitnessSol > fitness[index] and should_minimize == True) or (FitnessSol <= fitness[index] and should_minimize == False):
                trial[index] = 0
                foods[index][:] = np.copy(solution)
                f[index] = ObjValSol
                fitness[index] = FitnessSol
            else:
                trial[index] = trial[index] + 1
        iter_num+=1
        i += 1
        i = i % len(indexes)

class ABC:

    def __init__(_self, conf):
        _self.conf = conf
        _self.foods = np.zeros((_self.conf.FOOD_NUMBER, _self.conf.DIMENSION))
        _self.f = np.ones((_self.conf.FOOD_NUMBER))
        _self.fitness = np.ones((_self.conf.FOOD_NUMBER)) * np.iinfo(int).max
        _self.trial = np.zeros((_self.conf.FOOD_NUMBER))
        _self.prob = [0 for x in range(_self.conf.FOOD_NUMBER)]
        _self.solution = np.zeros((_self.conf.DIMENSION))
        _self.globalParams = [0 for x in range(_self.conf.DIMENSION)]
        _self.globalTime = 0
        _self.evalCount = 0
        _self.cycle = 0
        _self.experimentID = 0
        _self.globalOpts = list()
        if (_self.conf.MINIMIZE):
            _self.defualt_sol = 100, 0
        else:
            _self.defualt_sol = 0, 0

        if (_self.conf.SHOW_PROGRESS):
            _self.progressbar = progressbar.ProgressBar(max_value=_self.conf.MAXIMUM_EVALUATION)
        if (not(conf.RANDOM_SEED)):
            random.seed(conf.SEED)

    def calculate_function(_self, sol):
        try:
            if (_self.conf.SHOW_PROGRESS): 
                lock_eval.acquire()
                _self.progressbar.update(_self.evalCount)
                lock_eval.release()
            val = score_function(sol)
            return val
        except ValueError as err:
            print(err)
            print(
                "An exception occured: Upper and Lower Bounds might be wrong. (" + str(err) + " in calculate_function), values are: " + str(sol))
            return _self.defualt_sol
            # sys.exit()

    def calculate_fitness(_self, fun):
        _self.increase_eval()
        if fun >= 0:
            result = 1 / (fun + 1)
        else:
            result = 1 + abs(fun)
        return result
    
    def increase_evals(_self, n):
        lock_eval.acquire()
        _self.evalCount += n
        lock_eval.release()


    def increase_eval(_self):
        lock_eval.acquire()
        _self.evalCount += 1
        lock_eval.release()

    def stopping_condition(_self):
        lock_eval.acquire()
        status = bool(_self.evalCount >= _self.conf.MAXIMUM_EVALUATION)
        lock_eval.release()
        # status = status or bool(_self.conf.ietr_left <= 0)
        if(_self.conf.SHOW_PROGRESS):
          if(status == True and not( _self.progressbar._finished ) ):
               lock_progress.acquire()
               _self.progressbar.finish()
               lock_progress.release()
        return status

    def memorize_best_source(_self):
        for i in range(_self.conf.FOOD_NUMBER):
            if (_self.f[i] < _self.globalOpt and _self.conf.MINIMIZE == True) or (_self.f[i] >= _self.globalOpt and _self.conf.MINIMIZE == False):
                _self.globalOpt = np.copy(_self.f[i])
                _self.globalParams = np.copy(_self.foods[i][:])
    def get_param(_self, index):
        global_key = _self.conf.PARAMETERS_LIST[index]
        splited_key = global_key.split("/")
        class_name = splited_key[0]
        field_name = splited_key[1]
        return _self.conf.PARAMETERS_DICT[class_name][field_name]

    def get_upperbound(_self, index):
        return _self.get_param(index).maxBound

    def get_lowerbound(_self, index):
        return _self.get_param(index).minBound

    def get_new_parameter(_self, dim_index):
        idx = random.randint(0, len(_self.conf.PARAMS_POSSIBLE_VALUES[_self.conf.PARAMETERS_LIST[dim_index]]) - 1)
        val = _self.conf.PARAMS_POSSIBLE_VALUES[_self.conf.PARAMETERS_LIST[dim_index]][idx]
        match _self.get_param(dim_index).type:
            case "int":
                return int(val)
            case "float":
                trancuate = "%.1f" % val
                return float(trancuate) 
            case "double":
                trancuate = "%.1f" % val
                return float(trancuate) 
    
    def fix_new_params(_self, index):
        res, bad_params = _self.conf.CONSTRAINT_FUNC(_self.foods[index][:], _self.conf.PARAMETERS_LIST)
        while (not res):
            idx = random.randint(0, len(bad_params) - 1)
            param_idx = _self.conf.PARAMETERS_LIST.index(bad_params[idx])
            _self.foods[index][param_idx] = _self.get_new_parameter(param_idx)
            res, bad_params = _self.conf.CONSTRAINT_FUNC(_self.foods[index][:], _self.conf.PARAMETERS_LIST)

    def init(_self, index):
        if (not (_self.stopping_condition())):
            for i in range(_self.conf.DIMENSION):
                _self.foods[index][i] = _self.get_new_parameter(i)
            _self.fix_new_params(index)
            _self.solution = np.copy(_self.foods[index][:])
            res = _self.calculate_function(_self.solution)
            _self.f[index] = res[0]
            if (res[1]):
                _self.conf.ietr_left = 100
            else:
                _self.conf.ietr_left -= 1
            _self.fitness[index] = _self.calculate_fitness(_self.f[index])
            _self.trial[index] = 0

    def initial(_self):
        for i in range(_self.conf.FOOD_NUMBER):
            _self.init(i)
        _self.globalOpt = np.copy(_self.f[0])
        _self.globalParams = np.copy(_self.foods[0][:])

    def combine_solution(_self, sol_idx, foods, param2change, neighbour):
        r = random.random()
        val =  foods[sol_idx][param2change] + (
                        foods[sol_idx][param2change] - foods[neighbour][param2change]) * (
                                                             r - 0.5) * 2
        closest_value = min(_self.conf.PARAMS_POSSIBLE_VALUES[_self.conf.PARAMETERS_LIST[param2change]], key=lambda num: abs(num - val))
        return closest_value

    def get_updated_param(_self, sol_idx, foods, param2change, neighbour):
        closest_value = _self.combine_solution(sol_idx, foods, param2change, neighbour)
        match _self.get_param(param2change).type:
            case "int":
                closest_value = int(closest_value)
            case "float":
                trancuate = "%.1f" % closest_value
                closest_value = float(trancuate)  
            case "double":
                trancuate = "%.1f" % closest_value
                closest_value = float(trancuate) 
        return closest_value

    def update_param_multithread(_self, sol_idx, foods, param2change, neighbour):
        closest_value = _self.get_updated_param(sol_idx, foods, param2change, neighbour)
        solution = np.copy(foods[sol_idx][:])
        solution[param2change] = closest_value
        res, _ = _self.conf.CONSTRAINT_FUNC(solution, _self.conf.PARAMETERS_LIST)
        while (not res):
            closest_value = _self.get_updated_param(sol_idx, foods, param2change, neighbour)
            solution[param2change] = closest_value
            res, _ = _self.conf.CONSTRAINT_FUNC(solution, _self.conf.PARAMETERS_LIST)
        return closest_value



    def update_param(_self, sol_idx):

        closest_value = _self.update_param_multithread(sol_idx, _self.foods, _self.param2change, _self.neighbour) #TODO: If it works like that, not safe was not checked 
        match _self.get_param( _self.param2change).type:
            case "int":
                _self.solution[_self.param2change] = int(closest_value)
            case "float":
                trancuate = "%.1f" % closest_value
                _self.solution[_self.param2change] = float(trancuate)  
            case "double":
                trancuate = "%.1f" % closest_value
                _self.solution[_self.param2change] = float(trancuate) 

    def send_employed_bees_multithread(_self, indexes, foods):
        i = 0
        import ipdb; ipdb.set_trace()
        while (i < len(indexes)) and (not (_self.stopping_condition())):
            index = indexes[i]
            r = random.random()
            param2change = (int)(r * _self.conf.DIMENSION)

            r = random.random()
            neighbour = (int)(r * _self.conf.FOOD_NUMBER)
            while neighbour == index:
                r = random.random()
                neighbour = (int)(r * _self.conf.FOOD_NUMBER)
            solution = np.copy(foods[index][:])

            r = random.random()

            solution[param2change] = _self.update_param_multithread(index, foods, param2change, neighbour)
            ObjValSol = _self.calculate_function(solution)[0]
            FitnessSol = _self.calculate_fitness(ObjValSol)
            lock.acquire()
            if (FitnessSol > _self.fitness[index] and _self.conf.MINIMIZE == True) or (FitnessSol <= _self.fitness[index] and _self.conf.MINIMIZE == False):
                _self.trial[index] = 0
                _self.foods[index][:] = np.copy(solution)
                if (ObjValSol == 0):
                    import ipdb; ipdb.set_trace()
                _self.f[index] = ObjValSol
                _self.fitness[index] = FitnessSol
            else:
                _self.trial[index] = _self.trial[index] + 1
            lock.release()
            i += 1

    def send_employed_bees(_self):
        i = 0
        while (i < _self.conf.FOOD_NUMBER) and (not (_self.stopping_condition())):
            r = random.random()
            _self.param2change = (int)(r * _self.conf.DIMENSION)

            r = random.random()
            _self.neighbour = (int)(r * _self.conf.FOOD_NUMBER)
            while _self.neighbour == i:
                r = random.random()
                _self.neighbour = (int)(r * _self.conf.FOOD_NUMBER)
            _self.solution = np.copy(_self.foods[i][:])

            r = random.random()
            _self.update_param(i)
            if _self.solution[_self.param2change] < _self.get_lowerbound(_self.param2change):
                _self.solution[_self.param2change] = _self.get_lowerbound(_self.param2change)
            if _self.solution[_self.param2change] > _self.get_upperbound(_self.param2change):
                _self.solution[_self.param2change] = _self.get_upperbound(_self.param2change)
            _self.ObjValSol = _self.calculate_function(_self.solution)[0]
            _self.FitnessSol = _self.calculate_fitness(_self.ObjValSol)
            if (_self.FitnessSol > _self.fitness[i] and _self.conf.MINIMIZE == True) or (_self.FitnessSol <= _self.fitness[i] and _self.conf.MINIMIZE == False):
                _self.trial[i] = 0
                _self.foods[i][:] = np.copy(_self.solution)
                _self.f[i] = _self.ObjValSol
                _self.fitness[i] = _self.FitnessSol
            else:
                _self.trial[i] = _self.trial[i] + 1
            i += 1

    def calculate_probabilities(_self):
        maxfit = np.copy(max(_self.fitness))
        for i in range(_self.conf.FOOD_NUMBER):
            _self.prob[i] = (0.9 * (_self.fitness[i] / maxfit)) + 0.1
    
    def send_onlooker_bees_multithread(_self, indexes, foods):
        i = 0
        t = 0
        while (t < len(indexes)) and (not (_self.stopping_condition())):
            index = indexes[i]
            r = random.random()
            if ((r < _self.prob[index] and _self.conf.MINIMIZE == True) or (r > _self.prob[index] and _self.conf.MINIMIZE == False)):
                t+=1
                r = random.random()
                param2change = (int)(r * _self.conf.DIMENSION)
                r = random.random()
                neighbour = (int)(r * _self.conf.FOOD_NUMBER)
                while neighbour == index:
                    r = random.random()
                    neighbour = (int)(r * _self.conf.FOOD_NUMBER)
                solution = np.copy(foods[index][:])

                solution[param2change] = _self.update_param_multithread(index, foods, param2change, neighbour)
                ObjValSol = _self.calculate_function(solution)[0]
                FitnessSol = _self.calculate_fitness(ObjValSol)
                lock.acquire()
                if (FitnessSol > _self.fitness[index] and _self.conf.MINIMIZE == True) or (FitnessSol <= _self.fitness[index] and _self.conf.MINIMIZE == False):
                    _self.trial[index] = 0
                    _self.foods[index][:] = np.copy(solution)
                    if (ObjValSol == 0):
                        import ipdb; ipdb.set_trace()
                    _self.f[index] = ObjValSol
                    _self.fitness[index] = FitnessSol
                else:
                    _self.trial[index] = _self.trial[index] + 1
                lock.release()
            i += 1
            i = i % len(indexes)

    def send_onlooker_bees(_self):
        i = 0
        t = 0
        while (t < _self.conf.FOOD_NUMBER) and (not (_self.stopping_condition())):
            r = random.random()
            if ((r < _self.prob[i] and _self.conf.MINIMIZE == True) or (r > _self.prob[i] and _self.conf.MINIMIZE == False)):
                t+=1
                r = random.random()
                _self.param2change = (int)(r * _self.conf.DIMENSION)
                r = random.random()
                _self.neighbour = (int)(r * _self.conf.FOOD_NUMBER)
                while _self.neighbour == i:
                    r = random.random()
                    _self.neighbour = (int)(r * _self.conf.FOOD_NUMBER)
                _self.solution = np.copy(_self.foods[i][:])

                r = random.random()
                _self.update_param(i)
                if _self.solution[_self.param2change] < _self.get_lowerbound(_self.param2change):
                    _self.solution[_self.param2change] = _self.get_lowerbound(_self.param2change)
                if _self.solution[_self.param2change] > _self.get_upperbound(_self.param2change):
                    _self.solution[_self.param2change] = _self.get_upperbound(_self.param2change)

                _self.ObjValSol = _self.calculate_function(_self.solution)[0]
                _self.FitnessSol = _self.calculate_fitness(_self.ObjValSol)
                if (_self.FitnessSol > _self.fitness[i] and _self.conf.MINIMIZE == True) or (_self.FitnessSol <= _self.fitness[i] and _self.conf.MINIMIZE == False):
                    _self.trial[i] = 0
                    _self.foods[i][:] = np.copy(_self.solution)
                    _self.f[i] = _self.ObjValSol
                    _self.fitness[i] = _self.FitnessSol
                else:
                    _self.trial[i] = _self.trial[i] + 1
            i += 1
            i = i % _self.conf.FOOD_NUMBER

    def send_scout_bees(_self):
        if np.amax(_self.trial) >= _self.conf.LIMIT:
            _self.init(_self.trial.argmax(axis = 0))

    def increase_cycle(_self):
        _self.globalOpts.append(_self.globalOpt)
        _self.cycle += 1
    def setExperimentID(_self,run,t):
        _self.experimentID = t+"-"+str(run)
