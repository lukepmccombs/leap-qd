from leap_ec.ops import iteriter_op, listlist_op
from leap_ec.util import wrap_curry

from typing import Iterator, List, Callable

@wrap_curry
@iteriter_op
def assign_iterator_fitnesses(next_individual: Iterator, func: Callable=lambda x: x):
    """
    An operator that applies a function to each Individual's evaluation to
    produce its fitness. Individuals must be supplied from an iterator.

    Defaults to the identity function.
    """
    while True:
        ind = next(next_individual)
        ind.fitness = func(ind.evaluation)
        yield ind

@wrap_curry
@listlist_op
def assign_population_fitnesses(population: List, func: Callable=lambda x: x):
    """
    An operator that applies a function to each Individual's evaluation to
    produce its fitness. Must be supplied as a population.

    Defaults to the identity function.
    """
    for ind in population:
        ind.fitness = func(ind.evaluation)
    return population
