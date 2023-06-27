from leap_ec.ops import iteriter_op, listlist_op
from typing import Iterator
import torch
import torch.nn.functional as F
import numpy as np

from leap_qd.ops import iter_functional_fitness, list_functional_fitness


class BRNS:

    def __init__(
                self,
                random_encoder, trained_encoder, trained_optimizer,
                train_batch_size=64, error_function=F.mse_loss,
                concat_quality=False, evaluate_batch=False
            ):
        self.random_encoder = random_encoder
        self.trained_encoder = trained_encoder
        self.trained_optimizer = trained_optimizer
        self.train_batch_size = train_batch_size
        self.error_function = error_function
        
        self.concat_quality = concat_quality
        self.evaluate_batch = evaluate_batch

        self.dataset = []
    
    def add_evaluation(self, individual):
        self.dataset.append(individual.evaluation)

    def iter_add_evaluations_op(self, next_individual):
        @iteriter_op
        def _call(ni):
            while True:
                ind = next(ni)
                self.add_evaluation(ind)
                yield ind
        return _call(next_individual)

    def list_add_evaluations_op(self, population):
        @listlist_op
        def _call(pop):
            for ind in pop:
                self.add_evaluation(ind)
            return pop
        return _call(population)
    
    def clear_dataset(self):
        self.dataset.clear()
    
    def clear_dataset_op(self, arg):
        self.clear_dataset()
        return arg

    def train(self):
        self.random_encoder.train() # We put random_encoder in train mode in case of things like batch_norm
        self.trained_encoder.train()

        dataset_tensor = torch.tensor(np.array([dat["behavior"] for dat in self.dataset])).float()
        idx = torch.randperm(dataset_tensor.shape[0])
        for batch in torch.split(dataset_tensor[idx], self.train_batch_size):
            with torch.no_grad():
                random_encoding = self.random_encoder(batch)
            trained_encoding = self.trained_encoder(batch)
            loss = self.error_function(trained_encoding, random_encoding)

            self.trained_optimizer.zero_grad()
            loss.backward()
            self.trained_optimizer.step()

    def train_op(self, arg):
        self.train()
        return arg
    
    def evaluation_to_fitness(self, evaluation):
        behavior = evaluation["behavior"]
        behavior_tensor = torch.tensor(np.array(behavior)).float().unsqueeze(0)

        self.random_encoder.eval()
        self.trained_encoder.eval()

        with torch.no_grad():
            random_encoding = self.random_encoder(behavior_tensor)
            trained_encoding = self.trained_encoder(behavior_tensor)
            error = F.mse_loss(trained_encoding, random_encoding)

        fitness = error.detach().cpu().numpy()
        if self.concat_quality:
            quality = evaluation["quality"]
            if self.evaluate_batch:
                quality = np.mean(quality, axis=0)

            quality = np.array(quality)            
            if quality.ndim == 0:
                quality = quality.reshape((1,))
            
            fitness = np.concatenate([[fitness], quality])
            
        return fitness

    @property
    def iter_evaluation_to_fitness_op(self):
        return iter_functional_fitness(func=self.evaluation_to_fitness)
    
    @property
    def list_evaluation_to_fitness_op(self):
        return list_functional_fitness(func=self.evaluation_to_fitness)
    
