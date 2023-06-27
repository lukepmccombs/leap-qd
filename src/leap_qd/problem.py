from leap_ec.multiobjective.problems import MultiObjectiveProblem
import numpy as np


class AdaptiveMultiObjectiveProblem(MultiObjectiveProblem):
    """
    A version of MultiObjectiveProblem where maximize can be changed aposteriori.
    """

    def __init__(self, maximize=[True]):
        super.__init__(maximize)
    
    def set_maximize(self, maximize):
        self.maximize = np.where(maximize, 1, -1)