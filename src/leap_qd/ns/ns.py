from leap_ec.ops import iteriter_op, listlist_op
from leap_qd.ops import assign_iterator_fitnesses, assign_population_fitnesses
from leap_ec.util import wrap_curry


def dict_key_behavior(evaluation_dict):
    return evaluation_dict["behavior"]


class NoveltySearch:

    def __init__(self, archive):
        self.archive = archive
    
    @wrap_curry
    def add_iterator_evaluations(self, next_individual, key=dict_key_behavior):
        while True:
            ind = next(next_individual)
            self.archive.push(key(ind.evaluation))
            yield ind
    
    @wrap_curry
    def add_population_evaluations(self, population, key=dict_key_behavior):
        self.archive.extend([key(ind.evaluation) for ind in population])
        return population

    @wrap_curry
    def evaluation_to_fitness(self, evaluation, key=dict_key_behavior):
        return self.archive.novelty(key(evaluation))
    
    @wrap_curry
    def assign_iterator_fitnesses(self, next_individual, key=dict_key_behavior):
        return assign_iterator_fitnesses(next_individual, func=self.evaluation_to_fitness(key=key))
    
    @wrap_curry
    def assign_population_fitnesses(self, population, key=dict_key_behavior):
        return assign_population_fitnesses(population, func=self.evaluation_to_fitness(key=key))

    def save(self, checkpoint_fp):
        # This is just a wrapper around querying the archive, so forward
        self.archive.save(checkpoint_fp)
    
    def load(self, checkpoint_fp):
        # Same as with save
        self.archive.load(checkpoint_fp)