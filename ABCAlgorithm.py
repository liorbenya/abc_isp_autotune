# github page: https://abcolony.github.io/
# github repository: https://github.com/abcolony/ABCPython

import datetime
import sys
import time
import ABC
import math
import Config
from Reporter import Reporter
from threading import Thread, Lock

def run_workers(abc, func):
    threads = []
    for i in range(abc.conf.WORKERS):
        jump = math.floor(abc.conf.FOOD_NUMBER/abc.conf.WORKERS)
        indexes = range(jump * i, min(jump * (i+1), abc.conf.FOOD_NUMBER)) 
        foods = abc.foods[:]
        t = Thread(target=func, args=(indexes, foods))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

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
        # import ipdb; ipdb.set_trace()
        while(not(abc.stopping_condition())):
            # import ipdb; ipdb.set_trace()
            run_workers(abc, abc.send_employed_bees_multithread)
            # abc.send_employed_bees()
            abc.calculate_probabilities()
            # abc.send_onlooker_bees()
            run_workers(abc, abc.send_onlooker_bees_multithread)
            # send_onlooker_bees_multithread(_self, indexes, foods)
            abc.memorize_best_source()
            abc.send_scout_bees()
            abc.increase_cycle()

        abc.globalTime = time.time() * 1000 - start_time
        abcList.append(abc)
    Reporter(abcList)


if __name__ == '__main__':
    main(sys.argv[1:])
