import numpy as np
from collections import deque
from heapq import nsmallest
from scipy.spatial.distance import euclidean

import numpy as np
import abc
import pickle


class Archive(abc.ABC):
    
    @abc.abstractmethod
    def novelty(self, behavior):
        pass
    
    @abc.abstractmethod
    def push(self, behavior):
        pass

    @abc.abstractmethod
    def extend(self, behaviors):
        return [self.push(b) for b in behaviors]
    
    def save(self, checkpoint_fp):
        with open(checkpoint_fp, "wb") as c_file:
            pickle.dump(self, c_file)
    
    def load(self, checkpoint_fp):
        with open(checkpoint_fp, "rb") as c_file:
            new_self = pickle.load(c_file)
        self.__dict__.update(new_self.__dict__)
        

class NearestNeighborsArchive(Archive):

    def __init__(self, k, archive_size=None, distance_func=euclidean):
        self.archive_size = archive_size
        self.behaviors = deque(maxlen=archive_size)
        self.distance_func = distance_func
        self.k = k

    def novelty(self, behavior):


        return np.mean(
                nsmallest(self.k, [self.distance_func(behavior, ab) for ab in self.behaviors])
            )
    
    def push(self, behavior):
        self.behaviors.append(behavior)


class AdaptiveThresholdArchive(NearestNeighborsArchive):

    def __init__(
                self,
                k, min_novelty, check_period,
                relax_threshold=0, relax_rate=0.95,
                tighten_threshold=4, tighten_rate=1.2,
                unconditional_probability=0.001,
                archive_size=None, distance_func=euclidean,
            ):
        super().__init__(k, archive_size, distance_func)
        assert relax_threshold <= tighten_threshold,\
            "relax and tighten thresholds cannot overlap."
        
        self.min_novelty = min_novelty
        self.check_period = check_period
        self.relax_threshold = relax_threshold
        self.relax_rate = relax_rate
        self.tighten_threshold = tighten_threshold
        self.tighten_rate = tighten_rate
        self.unconditional_probability = unconditional_probability

        self._counter = 0
        self._num_passes = 0

    def push(self, behavior):
        self._counter += 1
        if self._counter > self.check_period:
            self._counter = 0
            self._num_passes = 0

            if self._num_passes <= self.relax_threshold:
                self.min_novelty *= self.relax_rate
            if self._num_passes > self.tighten_threshold:
                self.min_novelty *= self.tighten_rate

        if self.novelty(behavior) < self.min_novelty and\
            np.random.random() > self.unconditional_probability:
            return False
        
        self._num_passes += 1
        return super().push(behavior)


class GrowthRateArchive(NearestNeighborsArchive):

    def __init__(self, k, growth_rate, add_policy="novel", remove_policy="age", archive_size=None, distance_func=euclidean):
        super().__init__(k, archive_size, distance_func)
        assert add_policy in ("novel", "random")
        assert remove_policy in ("age", "random")

        self.growth_rate = growth_rate
        self.add_policy = add_policy
        self.remove_policy = remove_policy
    
    def push(self, behavior):
        raise NotImplementedError("GrowthRateArchive only accepts lists of behaviors")
    
    def extend(self, behaviors):
        if self.remove_policy == "random" and self.archive_size is not None:
            excess = len(self.behaviors) + min(self.growth_rate, len(behaviors)) - self.archive_size
            for _ in range(excess): # If zero or negative does nothing
                # Rotate a random number such that all indices have an equal probability of being next in queue
                # Centering just slightly reduces the number of ops
                shift = np.random.choice(len(self.behaviors)) - self.behaviors // 2
                self.behaviors.rotate(shift)
                self.behaviors.popleft()
        
        if self.add_policy == "novel":
            # Looking for maximum, so novelty is inverted
            choice_idx = np.argpartition([-self.novelty(b) for b in behaviors], self.growth_rate)
        else:
            choice_idx = np.random.choice(len(behaviors), self.growth_rate, replace=False)

        for idx in choice_idx:
            self.behaviors.append(behaviors[idx])