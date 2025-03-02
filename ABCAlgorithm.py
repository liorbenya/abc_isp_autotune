# github page: https://abcolony.github.io/
# github repository: https://github.com/abcolony/ABCPython

import datetime
import sys
import time
import ABC
import math
import Config
from Reporter import Reporter
import multiprocessing
import copy
import ipdb

## TODO: TRY FIX THIS
def run_workers_second(abc, func):
    processes = []
    lst_params = abc.conf.PARAMETERS_LIST
    dict_possiable_val = abc.conf.PARAMS_POSSIBLE_VALUES
    parameters_dict = abc.conf.PARAMETERS_DICT
    fitnesses_lst = [copy.deepcopy(abc.fitness) for _ in range(abc.conf.WORKERS)]
    trial_lst = [copy.deepcopy(abc.trial) for _ in range(abc.conf.WORKERS)]
    f_lst = [copy.deepcopy(abc.f) for _ in range(abc.conf.WORKERS)]
    prob_lst = [copy.deepcopy(abc.prob) for _ in range(abc.conf.WORKERS)]
    indexes_lst = []
    foods_list = [copy.deepcopy(abc.foods) for _ in range(abc.conf.WORKERS)]
    max_iteration_list = [copy.deepcopy(abc.conf.MAXIMUM_EVALUATION) for _ in range(abc.conf.WORKERS)]
    dim_lst =  [copy.deepcopy(abc.conf.DIMENSION) for _ in range(abc.conf.WORKERS)]
    food_number_lst = [copy.deepcopy(abc.conf.FOOD_NUMBER) for _ in range(abc.conf.WORKERS)]
    lst_params_lst = [copy.deepcopy(abc.conf.PARAMETERS_LIST) for _ in range(abc.conf.WORKERS)]
    dict_possiable_val_lst = [copy.deepcopy(dict_possiable_val) for _ in range(abc.conf.WORKERS)]
    parameters_dict = [copy.deepcopy(parameters_dict) for _ in range(abc.conf.WORKERS)]
    should_minimize_lst = [copy.deepcopy(abc.conf.MINIMIZE) for _ in range(abc.conf.WORKERS)]
    for i in range(abc.conf.WORKERS):
        food_number = abc.conf.FOOD_NUMBER
        jump = math.floor(food_number/abc.conf.WORKERS)
        indexes = range(jump * i, min(jump * (i+1), food_number)) 
        indexes_lst = indexes_lst + [indexes]
        # foods = abc.foods[:]
        # max_iterations = abc.conf.MAXIMUM_EVALUATION
        # dimansions = abc.conf.DIMENSION
        # should_minimize = abc.conf.MINIMIZE
        # p = multiprocessing.Process(target=func, args=(indexes, foods, max_iterations, dimansions, food_number, lst_params, dict_possiable_val, parameters_dict, fitnesses_lst[i], should_minimize, trial_lst[i], f_lst[i], prob_lst[i]))
        # processes.append(p)
        # p.start()
    result = zip(indexes_lst, foods_list, max_iteration_list, dim_lst, food_number_lst, lst_params_lst, dict_possiable_val_lst, parameters_dict, fitnesses_lst, should_minimize_lst, trial_lst, f_lst, prob_lst)
    with multiprocessing.Pool(processes=abc.conf.WORKERS) as pool:
        pool.starmap(func, result)


    # for p in processes:
    #     p.join()
    for i, indices in enumerate(indexes_lst):
        for idx in indices:
            abc.fitness[idx]  = fitnesses_lst[i][idx]
            abc.trial[idx] = trial_lst[i][idx]
            abc.f[idx] = f_lst[i][idx]
            abc.prob[idx] = prob_lst[i][idx]


def run_workers(abc, func):
    processes = []
    lst_params = abc.conf.PARAMETERS_LIST
    dict_possiable_val = abc.conf.PARAMS_POSSIBLE_VALUES
    parameters_dict = abc.conf.PARAMETERS_DICT
    # fitnesses_lst = [copy.deepcopy(abc.fitness) for _ in range(abc.conf.WORKERS)]
    # trial_lst = [copy.deepcopy(abc.trial) for _ in range(abc.conf.WORKERS)]
    # f_lst = [copy.deepcopy(abc.f) for _ in range(abc.conf.WORKERS)]
    # prob_lst = [copy.deepcopy(abc.prob) for _ in range(abc.conf.WORKERS)]
    indexes_lst = []
    # ipdb.set_trace()
    manager = multiprocessing.Manager()
    fitness_list = manager.list(abc.fitness)
    trial_lst = manager.list(abc.trial)
    f_lst = manager.list(abc.f)
    prob_lst = manager.list(abc.prob)
    for i in range(abc.conf.WORKERS):
        food_number = abc.conf.FOOD_NUMBER
        jump = math.floor(food_number/abc.conf.WORKERS)
        indexes = range(jump * i, min(jump * (i+1), food_number)) 
        indexes_lst = indexes_lst + [list(indexes)]
        foods = abc.foods[:]
        max_iterations = abc.conf.MAXIMUM_EVALUATION
        dimansions = abc.conf.DIMENSION
        should_minimize = abc.conf.MINIMIZE
        p = multiprocessing.Process(target=func, args=(indexes, foods, max_iterations, dimansions, food_number, lst_params, dict_possiable_val, parameters_dict, fitness_list, should_minimize, trial_lst, f_lst, prob_lst))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    # ipdb.set_trace()
    abc.fitness = list(fitness_list)
    abc.trial = list(trial_lst)
    abc.f = list(f_lst)
    abc.pron = list(prob_lst)

def main(argv):

    abcConf = Config.Config(argv)
    abcList = list()
    expT = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(" ","").replace(":","")
    for run in range(abcConf.RUN_TIME):

        abc = ABC.ABC(abcConf)
        abc.setExperimentID(run,expT)
        start_time = time.time() * 1000
        # import ipdb; ipdb.set_trace()
        abc.initial()
        abc.memorize_best_source()
        while(not(abc.stopping_condition())):
            # import ipdb; ipdb.set_trace()
            run_workers(abc, ABC.send_employed_bees_multprocess)
            # abc.send_employed_bees()
            abc.calculate_probabilities()
            # abc.send_onlooker_bees()
            run_workers(abc, ABC.send_onlooker_bees_multiprocess)
            # send_onlooker_bees_multithread(_self, indexes, foods)
            abc.memorize_best_source()
            abc.send_scout_bees()
            abc.increase_cycle()
            abc.increase_evals(abc.conf.WORKERS*abc.conf.MAXIMUM_EVALUATION)

        abc.globalTime = time.time() * 1000 - start_time
        abcList.append(abc)
    Reporter(abcList)


if __name__ == '__main__':
    main(sys.argv[1:])
