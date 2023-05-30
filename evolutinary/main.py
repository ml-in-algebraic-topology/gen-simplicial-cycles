import time
import warnings

from numpy.random import random, randint, choice
from numpy import ones

from freegroup.tools import (
    reduce_modulo_singleton_normal_closure, normalize, to_string,
    is_from_singleton_normal_closure, wu_closure
)
from freegroup.sampling import (
    freegroup_generator,
    normal_closure_generator,
)


def distance_to_singleton_normal_closure(word, closure, approximation="reduction"):
    if approximation == "reduction":
        return len(reduce_modulo_singleton_normal_closure(word, closure))
    else:
        raise NotImplementedError('unknown `approximation`')


def better_base(word, fdim, previous_function, first=None):
    if first:
        current_function = 0
        for idx in range(1, first + 1):
            current_function += distance_to_singleton_normal_closure(word, wu_closure(fdim, idx))
            if current_function >= previous_function:
                return False
        return True

    current_function = 0
    for idx in range(1, fdim + 1):
        current_function += distance_to_singleton_normal_closure(word, wu_closure(fdim, idx))
        if current_function >= previous_function:
            return False
    current_function += distance_to_singleton_normal_closure(word, wu_closure(fdim, 0))
    if current_function >= previous_function:
        return False
    return True


def dist_base(word, fdim, first=None):
    if first:
        return sum(distance_to_singleton_normal_closure(word, wu_closure(fdim, idx)) for idx in range(1, first + 1))
    return sum(distance_to_singleton_normal_closure(word, wu_closure(fdim, idx)) for idx in range(0, fdim + 1))


def optimize(
    word, dist, better, mutation_rate=0.1, generators_number=2, 
    max_iters=10, method='gemmate', fixed_size=False, verbose=True):

    # https://arxiv.org/pdf/1703.03334.pdf 3.2 (1 + 1) EA
    # https://arxiv.org/pdf/1812.11061.pdf 2.2 (\mu + \lambda) EA

    if method == 'gemmate' and fixed_size:
        warnings.warn('gemmate mutation method is not compatible with `fixed_size` set to True')

    generators = set(range(1, generators_number + 1)) | set(range(-generators_number, 0))
    def mutate(word, method='gemmation'):
        mutated_word = word.copy()
        if method == 'gemmate':
            i = randint(low = 0, high = len(word) + 1)
            if random() < mutation_rate:
                probas = ones((2 * generators_number + 1, ))
                probas[generators_number] = 0
                if i == len(word):
            	    probas[generators_number + mutated_word[i - 1]] = 0
                else:
                	probas[generators_number + mutated_word[i]] = 0
                	probas[generators_number + mutated_word[i - 1]] = 0
                to_insert = choice(2 * generators_number + 1, p = probas / probas.sum())
                mutated_word.insert(i, to_insert - generators_number)
            else:
                mutated_word.pop(min(i, len(word)-1))
        elif method == 'edit':
            for i in range(len(mutated_word)):
                if random() < mutation_rate:
                    probas = ones((2 * generators_number + 1, ))
                    probas[generators_number] = 0
               	    probas[mutated_words[i] + generators_number] = 0
                    mutated_word[i] = choice(
                		2 * generators_number + 1,
                		p = probas / probas.sum()
                	) - generators_number
        else:
            raise NotImplementedError('unknown `method`')
        return mutated_word

    current_function = dist(word)

    if verbose:
        print('INFO: optimization started')

    for _ in range(max_iters):
        new_word = mutate(word, method)
        normalized = normalize(new_word)

        if len(normalized) == 0:
            continue

        if better(normalized, current_function):
            word = (new_word if fixed_size else normalized).copy()
            current_function = dist(normalized)

            if verbose:
                print(f'INFO: f value = {current_function}')
                print(to_string(normalized, method = 'su'))

            if current_function == 0:
                break

    if verbose:
        print(
            f'INFO: optimization finished,', 
            'reached intersection' if current_function == 0 else 'reached max_iters', '\n'
            )

    return normalize(word), current_function == 0


class EvolutionarySampler:
    def __init__(
        self, generators_number=2, max_length=10, 
        exploration_rate=None, baseline="free", first=None, **kwargs):

        self.generators_number = generators_number
        self.max_length = max_length
        self.exploration_rate = exploration_rate

        if baseline == "free":
            self.baseline_group = freegroup_generator(
                generators_number, 'uh', {'radius': max_length})
        elif baseline == "joint":
            self.baseline_group = normal_closure_generator('conjugation', wu_closure(generators_number, 0),
            generators_number, 'uh', {'radius': max_length}, 'uh', {'radius': 5})
        elif baseline == "singleton":
            self.baseline_group = normal_closure_generator('conjugation', wu_closure(generators_number, 1), 
            generators_number, 'uh', {'radius': max_length}, 'uh', {'radius': 5})
        else:
            raise NotImplementedError('unknown `baseline`')

        if baseline in ["free", "joint", "singleton"]:
            self.dist = lambda word: dist_base(word, generators_number, first=first)
            self.better = lambda word, previous_function: better_base(word, generators_number, previous_function, first=first)
            if not first:
                self.condition = lambda word: all(
                    is_from_singleton_normal_closure(word, wu_closure(generators_number, idx)) 
                    for idx in range(0, generators_number + 1))
            else:
                self.condition = lambda word: all(
                    is_from_singleton_normal_closure(word, wu_closure(generators_number, idx)) 
                    for idx in range(1, first + 1))
        else:
            raise NotImplementedError()

        self.kwargs = kwargs

    def __iter__(self):
        return self

    def __next__(self):
        success = False
        while not success:
            word = next(self.baseline_group)
            if self.condition(word):
                return word
            if random() > self.exploration_rate:
                continue
            word, success = optimize(
                word, self.dist, self.better, 
                generators_number=self.generators_number, **self.kwargs)

        return word
    
    
from copy import deepcopy
from multiprocessing import Pool
from tqdm.auto import tqdm
from numpy.random import seed as set_seed, default_rng
from time import time
from argparse import ArgumentParser
from pandas import DataFrame

def run_experiment(config):
    sampler_kwargs = {k[len('sampler_'):]: v for k, v in config.items() if k.startswith('sampler_')}
    sampler = EvolutionarySampler(**sampler_kwargs)
    
    rng = default_rng(config['seed'])
    
    ratio = 0 
    for _ in range(config['num_samples']):
        word = next(sampler.baseline_group)
        for _ in range(config['num_tries']):
            set_seed(rng.integers(0, 1000))
            word, success = optimize(
                word, sampler.dist, sampler.better, 
                generators_number=sampler.generators_number, **sampler.kwargs)
            if success: ratio += 1; break
    result = deepcopy(config)
    result['completion_ratio'] = ratio / config['num_samples']
    return result

if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-s', '--seed', type=int, default = 42)
    p.add_argument('-t', '--num_tries', type=int, default = 5)
    p.add_argument('-N', '--num_samples', type=int, default = 1000)
    p.add_argument('-b', '--sampler_baseline', type=str, choices = ['free', 'singleton', 'join'], default='singleton')
    p.add_argument('-n', '--sampler_generators_number', type=int, default = 3)
    p.add_argument('-l', '--sampler_max_length', type=int, default = 60)
    p.add_argument('-i', '--sampler_max_iters', type=int, default = 400)
    p.add_argument('-r', '--sampler_mutation_rate', type=float, default=0.8)
    p.add_argument('-m', '--sampler_method', type=str, choices = ['gemmate', 'edit'], default='gemmate')
    p.add_argument('-v', '--sampler_verbose', type=bool, default=False)
    p.add_argument('-o', '--output', type=str, default='results.csv')

    args = p.parse_args()

    experiments = []

    for fdim, max_length, num_tries in [
        (3, 60, 5),
        (3, 60, 10),
        (3, 60, 20),
        (4, 100, 5),
        (4, 100, 10),
        (4, 100, 20),
        (5, 200, 5),
        (5, 200, 10),
        (5, 200, 20),    
    ]:
        kwargs = dict(args._get_kwargs())
        kwargs['sampler_generators_number'] = fdim
        kwargs['sampler_max_length'] = max_length
        kwargs['num_tries'] = num_tries
        kwargs.pop('output')
        
        experiments.append(kwargs)

    results = {}
    with Pool(5) as p:
        for result in tqdm(p.imap_unordered(run_experiment, experiments), total = len(experiments)):
            for k, v in result.items():
                if not k in results: results[k] = []
                results[k].append(v)
    DataFrame(results).to_csv(args.output, index=False)

