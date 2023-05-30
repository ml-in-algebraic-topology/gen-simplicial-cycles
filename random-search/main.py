from numpy.random import randint, shuffle
from freegroup.sampling import normal_closure, random_tree
from freegroup.tools import (
    wu_closure, is_from_singleton_normal_closure,
    flatten, normalize
)

def incomplete_intersection(
    fdim: int,
    zero_closure_length_method: str,
    zero_closure_length_radius: int,
    non_zero_closure_length_method: str,
    non_zero_closure_length_radius: int,
    normal_closure_method: str = 'brackets',
    normal_closure_kwargs = {},
):
    words = []
    
    exclude_idx = randint(0, fdim + 1)
    for idx in range(fdim + 1):
        if idx == exclude_idx: continue
        
        length_method = zero_closure_length_method if idx == 0 else non_zero_closure_length_method
        length_parameters = {'radius': zero_closure_length_radius if idx == 0 else non_zero_closure_length_radius}
        
        words.append(normal_closure(
            normal_closure_method,
            closure = wu_closure(fdim, idx), freegroup_dimension = fdim,
            depth_method = length_method, depth_parameters = length_parameters,
            **normal_closure_kwargs,
        ))
    
    shuffle(words)
    return normalize(flatten(random_tree(words)))


from argparse import ArgumentParser
from tqdm.auto import tqdm
from numpy.random import default_rng, seed as set_seed
from copy import deepcopy
from multiprocessing import Pool
from pandas import DataFrame

def run(config):
    result = deepcopy(config)
    set_seed(config['seed'])

    ratio = 0
    for _ in range(config['num_samples']):
        word = incomplete_intersection(
            fdim = config['fdim'],
            zero_closure_length_method = config['zero_closure_length_method'],
            zero_closure_length_radius = config['zero_closure_length_radius'],
            non_zero_closure_length_method = config['non_zero_closure_length_method'],
            non_zero_closure_length_radius = config['non_zero_closure_length_radius'],
            normal_closure_method = config['method'],
        )
        
        if all([
            is_from_singleton_normal_closure(word, wu_closure(config['fdim'], idx))
            for idx in range(config['fdim'] + 1)
        ]):
            ratio += 1
    
    result['completion_ratio'] = ratio / config['num_samples']
    return result
        
        

if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-n', '--fdim', type = int, default = 3)
    p.add_argument('-lm0', '--zero_closure_length_method', type = str, default = 'u')
    p.add_argument('-lr0', '--zero_closure_length_radius', type = int, default = 10)
    p.add_argument('-lmn0', '--non_zero_closure_length_method', type = str, default = 'u'),
    p.add_argument('-lrn0', '--non_zero_closure_length_radius', type = int, default = 30)
    p.add_argument('-m', '--method', type = str, choices = ['brackets', 'conjugate'], default = 'brackets')
    p.add_argument('-s', '--seed', type = int, default = 42)
    p.add_argument('-N', '--num_samples', type = int, default = 1000)
    p.add_argument('-j', '--threads', type = int, default = 1)
    p.add_argument('-o', '--output', type = str, default = 'results.csv')
    
    args = p.parse_args()
    
    experiments, results = [], {}
    
    for fdim, zclr, nzclr in [
        (3, 10, 30),
        (4, 8, 30),
        (5, 7, 30),
    ]:
        exp = dict(args._get_kwargs())
        del exp['output'], exp['threads']
        
        exp['fdim'] = fdim
        exp['zero_closure_length_radius'] = zclr
        exp['non_zero_closure_length_radius'] = nzclr
        
        experiments.append(exp)

    with Pool(args.threads) as pool:
        for result in tqdm(pool.imap_unordered(run, experiments), total = len(experiments)):
            for k, v in result.items():
                if not k in results: results[k] = []
                results[k].append(v)
                
    DataFrame(results).to_csv(args.output, index = False)
