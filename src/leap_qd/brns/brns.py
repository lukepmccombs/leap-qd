from leap_ec.ops import iteriter_op, listlist_op
import torch
import torch.nn.functional as F
import numpy as np
import os

from leap_qd.ops import assign_iterator_fitnesses, assign_population_fitnesses


class BRNS:

    def __init__(
                self,
                random_encoder, trained_encoder, trained_optimizer,
                concat_quality=False, evaluate_batch=False
            ):
        self.random_encoder = random_encoder
        self.trained_encoder = trained_encoder
        self.trained_optimizer = trained_optimizer
        
        self.concat_quality = concat_quality
        self.evaluate_batch = evaluate_batch

        self.dataset = []
    
    def add_evaluation(self, individual):
        self.dataset.append(individual.evaluation)

    def add_iterator_evaluations(self, next_individual):
        @iteriter_op
        def _call(ni):
            while True:
                ind = next(ni)
                self.add_evaluation(ind)
                yield ind
        return _call(next_individual)

    def add_population_evaluations(self, population):
        @listlist_op
        def _call(pop):
            for ind in pop:
                self.add_evaluation(ind)
            return pop
        return _call(population)
    
    def clear_dataset_(self):
        self.dataset.clear()
    
    def clear_dataset(self, arg):
        self.clear_dataset_()
        return arg

    def train_mode(self):
        self.random_encoder.train()
        self.trained_encoder.train()
    
    def eval_mode(self):
        self.random_encoder.eval()
        self.trained_encoder.eval()

    def train_(self, behaviors, inverted=False):
        self.train_mode()

        dataset_tensor = torch.tensor(np.array(behaviors)).float()
        loss = self.batch_behavior_error(dataset_tensor)
        if inverted:
            loss = loss * -1

        self.trained_optimizer.zero_grad()
        loss.backward()
        self.trained_optimizer.step()

    def batch_behavior_error(self, batch, reduction="mean"):
        with torch.no_grad():
            random_encoding = self.random_encoder(batch)
        trained_encoding = self.trained_encoder(batch)
        return F.mse_loss(trained_encoding, random_encoding, reduction=reduction)
    
    def pre_train(self, low, high, batch_size, iterations):
        for _ in range(iterations):
            behaviors = np.random.uniform(low, high, (batch_size, len(low)))
            self.train_(behaviors, True)

    def train(self, arg):
        self.train_([dat["behavior"] for dat in self.dataset])
        return arg
    
    def evaluation_to_fitness(self, evaluation):
        behavior = evaluation["behavior"]
        behavior_tensor = torch.tensor(np.array(behavior)).float().unsqueeze(0)

        self.eval_mode()

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

    def batch_evaluation_to_fitness(self, evaluations):
        behaviors = [e["behavior"] for e in evaluations]
        behavior_tensor = torch.tensor(np.array(behaviors)).float()
        self.eval_mode()

        with torch.no_grad():
            random_encoding = self.random_encoder(behavior_tensor)
            trained_encoding = self.trained_encoder(behavior_tensor)
            error = F.mse_loss(trained_encoding, random_encoding, reduction="none")
            error = torch.mean(error, 1)

        fitness = error.detach().cpu().numpy()
        if self.concat_quality:
            # Ignoring for now
            pass
            
        return fitness

    @property
    def assign_iterator_fitnesses(self):
        return assign_iterator_fitnesses(func=self.evaluation_to_fitness)
    
    @property
    def assign_population_fitnesses(self):
        return assign_population_fitnesses(func=self.evaluation_to_fitness)
    
    def save(self, checkpoint_fp):
        torch.save({
            "random_encoder_dict": self.random_encoder.state_dict(),
            "trained_encoder_dict": self.trained_encoder.state_dict(),
            "trained_optimizer_dict": self.trained_optimizer.state_dict(),
            "concat_quality": self.concat_quality,
            "evaluate_batch": self.evaluate_batch,
            "dataset": self.dataset
        }, checkpoint_fp)
    
    def load(self, checkpoint_fp):
        data = torch.load(checkpoint_fp)

        self.random_encoder.load_state_dict(data["random_encoder_dict"])
        self.trained_encoder.load_state_dict(data["trained_encoder_dict"])
        self.trained_optimizer.load_state_dict(data["trained_optimizer_dict"])
        self.concat_quality = data["concat_quality"]
        self.evaluate_batch = data["evaluate_batch"]
        self.dataset = data["dataset"]