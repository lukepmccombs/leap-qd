from leap_ec.individual import Individual as _Individual, RobustIndividual as _RobustIndividual
from leap_ec.distrib.individual import DistributedIndividual as _DistributedIndividual


class EvaluationOverrideMixin:
    """
    A mixin that wraps evaluate to redirect the assignment of fitness to evaluation
    instead.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluation = None

    def evaluate(self):
        fitness = self.fitness
        self.evaluation, self.fitness = super().evaluate(), fitness
        return self.evaluation


class EvaluationIndividual(EvaluationOverrideMixin, _Individual):
    """
    A modified version of the base Individual that stores the evaluation separately
    from the fitness. This allows the user to return any set of values from a problem
    and separately determine an Indiviudal's fitness. Notably, this is used for quality
    diversity algorithms, where the result of a problem is at minimum a behavior
    descriptor that must be translated into a valid fitness.

    >>> from leap_ec.decoder import IdentityDecoder
    >>> from leap_ec.binary_rep.problems import MaxOnes
    >>> import numpy as np
    >>> genome = np.ones(3)
    >>> decoder = IdentityDecoder()
    >>> problem = MaxOnes()
    >>> ind = EvaluationIndividual(genome, decoder, problem)
    >>> ind.evaluate()
    3
    >>> print(ind.evaluation)
    3
    >>> print(ind.fitness)
    None
    """

class RobustEvaluationIndividual(EvaluationOverrideMixin, _RobustIndividual):
    """
    A modified version of RobustIndividual that stores the evaluation separately
    from the fitness. This allows the user to return any set of values from a problem
    and separately determine an Indiviudal's fitness. Notably, this is used for quality
    diversity algorithms, where the result of a problem is at minimum a behavior
    descriptor that must be translated into a valid fitness.

    >>> from leap_ec.decoder import IdentityDecoder
    >>> from leap_ec.binary_rep.problems import MaxOnes
    >>> import numpy as np
    >>> genome = np.ones(3)
    >>> decoder = IdentityDecoder()
    >>> problem = MaxOnes()
    >>> ind = RobustEvaluationIndividual(genome, decoder, problem)
    >>> ind.evaluate()
    3
    >>> print(ind.evaluation)
    3
    >>> print(ind.fitness)
    None
    """
    

class DistributedEvaluationIndividual(EvaluationOverrideMixin, _DistributedIndividual): 
    """
    A modified version of DistributedIndividual that stores the evaluation separately
    from the fitness. This allows the user to return any set of values from a problem
    and separately determine an Indiviudal's fitness. Notably, this is used for quality
    diversity algorithms, where the result of a problem is at minimum a behavior
    descriptor that must be translated into a valid fitness.

    >>> from leap_ec.decoder import IdentityDecoder
    >>> from leap_ec.binary_rep.problems import MaxOnes
    >>> import numpy as np
    >>> genome = np.ones(3)
    >>> decoder = IdentityDecoder()
    >>> problem = MaxOnes()
    >>> ind = DistributedEvaluationIndividual(genome, decoder, problem)
    >>> ind.evaluate()
    3
    >>> print(ind.evaluation)
    3
    >>> print(ind.fitness)
    None
    """

