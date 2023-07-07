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
from leap_torch.ops import mutate_gaussian, UniformCrossover
from leap_qd.individual import EvaluationIndividual, DistributedEvaluationIndividual
from leap_qd.ns.ns import NoveltySearch
from leap_qd.ns.archive import AdaptiveThresholdArchive

import os
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt
from torch import nn
from itertools import pairwise, count
from distributed import Client


class BehaviorCartPole(ScalarProblem):
    """
    A cart pole problem that returns a dictionary containing a behavior
    descriptor of 8 values, a concatenation of the first and last observation.
    """

    def __init__(self, env, runs_per_eval):
        super().__init__(True)
        self.env = env
        self.runs_per_eval = runs_per_eval
    
    @property
    def behavior_size(self):
        return 8 * self.runs_per_eval
    
    def evaluate(self, phenome):
        all_behaviors = []

        for _ in range(self.runs_per_eval):
            obs, _ = self.env.reset()
            behavior = list(obs)

            for i in count():
                action = phenome(obs)
                obs, _, terminated, truncated, _ = self.env.step(action)

                if terminated or truncated:
                    behavior += list(obs)
                    behavior.append(i)
                    break
            
            all_behaviors.append(behavior)
        
        behavior = np.array(all_behaviors).flatten()
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
            runs_per_eval, simulation_steps,
            pop_size, num_generations,
            mutate_std, p_mutate,
            model_hidden_layer_size, model_num_hidden_layer,
            demo_modulo=10, num_workers=0
        ):
    
    env = gym.make("CartPole-v1", simulation_steps)
    demo_env = gym.make("CartPole-v1", simulation_steps, render_mode="human")
    prob = BehaviorCartPole(env, runs_per_eval)
    
    ns = NoveltySearch(AdaptiveThresholdArchive(5, 1, pop_size * 3))
    
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
                ns.assign_population_fitnesses,
                fitness_probe,
                demo_probe,
                ops.tournament_selection,
                ops.clone,
                mutate_gaussian(std=mutate_std, p_mutate=p_mutate),
                UniformCrossover(),
                *evaluate_pipeline,
                ns.assign_population_fitnesses,
                ns.add_population_evaluations
            ]
        )
    
    client.close()

if __name__ == "__main__":
    main(
        1, 500,
        100, 250,
        0.001, 1.0,
        10, 3,
        10,
        8
    )