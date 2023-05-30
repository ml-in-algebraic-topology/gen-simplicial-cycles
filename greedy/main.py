from freegroup.tools import (
    wu_closure, reduce_modulo_singleton_normal_closure,
    reciprocal, reduce_modulo_singleton_normal_closure_step,
    is_from_singleton_normal_closure
)

from tqdm.auto import tqdm

from numpy.random import choice
from numpy import zeros, flatnonzero, inf

def generate_from_intersection(
    prefix,
    fdim: int,
    max_iters: int,
    closure_point: int = 1,
    reduction_point: int = 3,
    verbose = False,
):
    generated = prefix[::]

    stacks = [reduce_modulo_singleton_normal_closure(prefix, wu_closure(fdim, idx))
                for idx in range(fdim + 1)]

    iteration = 0
    while not all(map(lambda v: len(v) == 0, stacks)) and iteration < max_iters:
        iteration += 1
        scores = zeros((2 * fdim + 1, ))
        
        for idx, stack in enumerate(stacks):
            if len(stack) == 0: continue
            
            base, _base = wu_closure(fdim, idx), reciprocal(wu_closure(fdim, idx))

            last = stack[-1]
            if last == generated[-1]:
                if idx == 0 and last < 0:
                    # X_n --> "+1" to X_{n-1}, ..., X_1 --> "+1" to X_n
                    scores[((abs(last) - 1) + fdim - 1) % fdim - (fdim + 1)] += closure_point
                if idx == 0 and last > 0:
                    # x_1 --> "+1" to x_2, ..., x_n --> "+1" to x_1
                    scores[((abs(last) - 1) + 1) % fdim + (fdim + 1)] += closure_point
                if idx > 0:
                    scores[idx + fdim] += closure_point
                    scores[-idx + fdim] += closure_point
            # add score to reduce stack
            scores[-last + fdim] += reduction_point
        
        # ensure that there is no reduction
        scores[-generated[-1] + fdim] = -inf
        scores[0 + fdim] = -inf
            
        key = choice(flatnonzero(scores == scores.max()))
        generated.append(key - fdim)
        
        for idx in range(fdim + 1):
            reduce_modulo_singleton_normal_closure_step(
                stacks[idx], generated[-1], wu_closure(fdim, idx)
            )
    
    return generated
        

from argparse import ArgumentParser
from pandas import DataFrame
from multiprocessing import Pool
from copy import deepcopy
from numpy.random import default_rng, seed as set_seed
from freegroup.sampling import freegroup

def run(config):
    result = deepcopy(config)

    rng = default_rng(config['seed'])
    ratio = 0
    for _ in range(config['num_samples']):
        prefix = freegroup(
            config['freegroup_dimension'],
            'c', {'radius': config['prefix_length']}
        )
        
        for _ in range(config['num_tries']):
            set_seed(rng.integers(0, 1000))
            
            generated = generate_from_intersection(
                prefix, fdim = config['freegroup_dimension'], max_iters = config['max_iters'],
                closure_point = config['closure_point'], reduction_point = config['reduction_point'])
                
            if all([
                is_from_singleton_normal_closure(generated, wu_closure(config['freegroup_dimension'], idx))
                for idx in range(config['freegroup_dimension'] + 1)
            ]):
                ratio += 1
    result['completion_ratio'] = ratio / config['num_samples']
    return result


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-n', '--freegroup_dimension', type = int, default = 3)
    p.add_argument('-T', '--max_iters', type = int, default = 100)
    p.add_argument('-p', '--prefix_length', type = int, default = 5)
    p.add_argument('-t', '--num_tries', type = int, default = 5)
    p.add_argument('-N', '--num_samples', type = int, default = 1000)
    p.add_argument('-s', '--seed', type = int, default = 42)
    p.add_argument('-o', '--output', type = str, default = 'results.csv')
    p.add_argument('-j', '--threads', type = int, default = 5)
    p.add_argument('-c', '--closure_point', type = int, default = 1)
    p.add_argument('-r', '--reduction_point', type = int, default = 3)
    
    args = p.parse_args()
    
    experiments, results = [], {}
    
    for fdim, max_iters in [
        (3, 50),
        (3, 100),
        (3, 150),
        (4, 100),
        (4, 200),
        (5, 200),
        (5, 250),
        (5, 300)
    ]:
        exp = dict(args._get_kwargs())
        del exp['output']
        del exp['threads']
        
        exp['freegroup_dimension'] = fdim
        exp['max_iters'] = max_iters
        experiments.append(exp)
    
    with Pool(args.threads) as pool:
        for result in tqdm(pool.imap_unordered(run, experiments), total = len(experiments)):
            for k, v in result.items():
                if not k in results: results[k] = []
                results[k].append(v)
    
    DataFrame(results).to_csv(args.output, index = False)

