from leap_ec.problem import ScalarProblem
from leap_ec.algorithm import generational_ea
from leap_ec.individual import Individual
from leap_ec.distrib.synchronous import eval_population
from leap_ec import ops
from leap_ec.executable_rep.executable import ArgmaxExecutable, WrapperDecoder
from leap_ec.executable_rep.problems import EnvironmentProblem
from leap_ec.representation import Representation
from leap_ec.probe import PopulationMetricsPlotProbe
from leap_ec.util import get_step
from leap_torch.initializers import create_instance
from leap_torch.decoders import NumpyDecoder
from leap_torch.ops import mutate_guassian, UniformCrossover
from leap_qd.individual import EvaluationIndividual, DistributedEvaluationIndividual
from leap_qd.brns.brns import BRNS

import os
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt
from torch import nn
from itertools import pairwise
from distributed import Client


class BehaviorCartPole(ScalarProblem):
    """
    A cart pole problem that returns a dictionary containing a behavior
    descriptor of 8 values, a concatenation of the first and last observation.

    If batch_mode is "mean", behaviors are averaged over runs. If "flat", they
    are flattened into an array of size runs * 8. If "stack", they are stacked
    along dimension 0.
    """

    def __init__(self, env, runs_per_eval, batch_mode="flat"):
        super().__init__(True)
        self.env = env
        self.runs_per_eval = runs_per_eval
        self.batch_mode = batch_mode
    
    @property
    def behavior_size(self):
        if self.batch_mode == "flat":
            return 8 * self.runs_per_eval
        return 8
    
    def evaluate(self, phenome):
        all_behaviors = []

        for _ in range(self.runs_per_eval):
            obs, _ = self.env.reset()
            behavior = list(obs)

            while True:
                action = phenome(obs)
                obs, _, terminated, truncated, _ = self.env.step(action)

                if terminated or truncated:
                    behavior += list(obs)
                    break
            
            all_behaviors.append(behavior)
        
        if self.batch_mode == "mean":
            behavior = np.mean(all_behaviors, axis=0)
        elif self.batch_mode == "flat":
            behavior = np.array(all_behaviors).flatten()
        elif self.batch_mode == "stack":
            behavior = np.array(all_behaviors)
        else:
            raise NotImplementedError(self.batch_mode)
        return {"behavior": behavior}


class SimpleModel(nn.Module):

    def __init__(self, num_inputs, hidden_layer_sizes, num_outputs):
        super().__init__()
        
        latent_sizes = [num_inputs, *hidden_layer_sizes, num_outputs]
        self.layers = nn.ModuleList([
                nn.Linear(*size) for size in pairwise(latent_sizes)
            ])
    
    def forward(self, inputs):
        latent = inputs
        for layer in self.layers:
            latent = layer(latent)
            latent = F.relu(latent)
        return latent
        

def main(
            runs_per_eval, simulation_steps, batch_mode,
            pop_size, num_generations,
            mutate_std, p_mutate,
            model_hidden_layer_size, model_num_hidden_layer,
            encoder_hidden_layer_size, random_num_layers, trained_num_layers, c_factor,
            demo_modulo=10, num_workers=0
        ):
    
    env = gym.make("CartPole-v1", simulation_steps)
    demo_env = gym.make("CartPole-v1", simulation_steps, render_mode="human")
    prob = BehaviorCartPole(env, runs_per_eval, batch_mode)
    
    behavior_size = prob.behavior_size
    encoding_size = int(np.ceil(behavior_size * c_factor))
    random_encoder = SimpleModel(behavior_size, (encoder_hidden_layer_size,) * random_num_layers, encoding_size)
    trained_encoder = SimpleModel(behavior_size, (encoder_hidden_layer_size,) * trained_num_layers, encoding_size)

    optimizer = torch.optim.RMSprop(trained_encoder.parameters(), lr=1e-2)
    
    brns = BRNS(random_encoder, trained_encoder, optimizer, evaluate_batch=batch_mode=="stack")
    
    evaluate_pipeline = [
            ops.pool(size=pop_size),
            Individual.evaluate_population,
        ]
    if num_workers > 0:
        client = Client(n_workers=num_workers)
        evaluate_pipeline = [
                ops.pool(size=pop_size),
                eval_population(client=client)
            ]
    
    decoder = WrapperDecoder(NumpyDecoder(), ArgmaxExecutable)
    fitness_probe = PopulationMetricsPlotProbe(
            ax=plt.gca(), metrics=[
                lambda pop: np.mean([ind.fitness for ind in pop]),
                lambda pop: np.max([ind.fitness for ind in pop])
            ], xlim=(0, num_generations)
        )
    
    def demo_probe(population):
        if get_step() % demo_modulo == 0:
            most_diverse = max(population)
            demo_prob = EnvironmentProblem(1, simulation_steps, demo_env, "reward", gui=True)
            demo_prob.evaluate(most_diverse.phenome)
        return population
    
    def save_probe(p):
        base_dir = f"./models/gen{get_step()}"
        os.makedirs(base_dir)
        torch.save(random_encoder, os.path.join(base_dir, "random_encoder.pt"))
        torch.save(trained_encoder, os.path.join(base_dir, "trained_encoder.pt"))
        torch.save(torch.tensor(np.array([ind.evaluation["behavior"] for ind in p])), os.path.join(base_dir, "population.pt"))
        return p

    final_pop = generational_ea(
            max_generations=num_generations, pop_size=pop_size,

            problem=prob,

            representation=Representation(
                initialize=create_instance(
                    SimpleModel,
                    env.observation_space.shape[-1], (model_hidden_layer_size,) * model_num_hidden_layer, env.action_space.n
                ),
                decoder=decoder,
                individual_cls=(EvaluationIndividual if num_workers > 0 else DistributedEvaluationIndividual),
            ),

            pipeline=[
                save_probe,
                brns.clear_dataset,
                brns.assign_population_fitnesses,
                brns.add_population_evaluations,
                fitness_probe,
                demo_probe,
                ops.tournament_selection,
                ops.clone,
                mutate_guassian(std=mutate_std, p_mutate=p_mutate),
                UniformCrossover(),
                *evaluate_pipeline,
                brns.assign_population_fitnesses,
                brns.add_population_evaluations,
                brns.train
            ]
        )
    
    client.close()

if __name__ == "__main__":
    main(
        1, 500, "flat",
        100, 250,
        0.001, 1.0,
        10, 3,
        24, 3, 5, 1,
        10,
        8
    )