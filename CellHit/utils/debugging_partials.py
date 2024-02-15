from functools import partial

def debug_search(func,drugID,mode='kwnoledge', dataset='gdsc',gpuID=0,random_state=0):
    fixed_args = {'n_trials':2, 'n_startup_trials':2, 'num_parallel_tree':1}
    
    if mode == 'kwnoledge':
        fixed_args['cv_iterations'] = 2

    def wrapper(*args,**kwargs):
        #ignore the kwargs and args
        return func(int(drugID), dataset=dataset, gpuID=int(gpuID), random_state=random_state, **fixed_args)
    return wrapper


