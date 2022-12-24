import math
import concurrent.futures
from multiprocessing import Manager
from src import generate_game, N_CTPS, evaluate, compute_traj, RADIUS, plot_game
from agent import Agent
from tqdm import tqdm  # a convenient progress bar
import torch
from itertools import product
import numpy as np

N_EVALS = 100

def param_gen(param_dict: dict):
    items = param_dict.items()
    keys, values = zip(*items)
    for v in product(*values):
        param = dict(zip(keys, v))
        yield param


if __name__ == "__main__":
    n_targets = 40


    # This is a example of what the evaluation procedure looks like.
    # The whole dataset is divided into a training set and a test set.
    # The training set (including `data` and `label`) is distributed to you.
    # But in the final evaluation we will use the test set.

    data = torch.load("data.pth")
    label = data["label"]
    feature = data["feature"]
    agent = Agent()

    param_dict = {
        "stdx": np.arange(1., 4., .1, dtype=np.float32),
        "stdy": np.arange(1, 2, .1, dtype=np.float32),
        "meanx": np.arange(-1., 2.5, .1, dtype=np.float32),
        "meany": np.arange(0, 1, .1, dtype=np.float32)
    }

    best_score = -math.inf
    best_param = None


    def evap(param, best_result):
        scores = []
        for _ in range(N_EVALS):
            target_pos, target_features, target_cls, class_scores = generate_game(n_targets, N_CTPS, feature, label)
            ctps_inter = agent.get_action(target_pos, target_features, class_scores, **param)
            score = evaluate(compute_traj(ctps_inter), target_pos, class_scores[target_cls], RADIUS)
            scores.append(score)
        score = torch.stack(scores).float().mean()

        return param, score

    manager = Manager()
    best_result = manager.dict({ "score": -math.inf, "param": None })
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     futures = [executor.submit(evap, param, best_result) for param in param_gen(param_dict)]
    #     for future in concurrent.futures.as_completed(futures):
    #         param, score = future.result()
    #
    #         if score > best_result["score"]:
    #             best_result["score"] = score
    #             best_result["param"] = param
    #
    #         print(f"Score: {score}")
    #         print(f"Param: {param}")
    #         print(f"Best score: {best_result['score']}")
    #         print(f"Best param: {best_result['param']}")
    #         print()
    # print(dict)


    for param in tqdm(param_gen(param_dict)):
        scores = []
        for _ in range(N_EVALS):
            target_pos, target_features, target_cls, class_scores = generate_game(n_targets, N_CTPS, feature, label)
            ctps_inter = agent.get_action(target_pos, target_features, class_scores, **param)
            score = evaluate(compute_traj(ctps_inter), target_pos, class_scores[target_cls], RADIUS)
            scores.append(score)
        mean_score = torch.stack(scores).float().mean()
        if score > best_result["score"]:
            best_result["score"] = score
            best_result["param"] = param
        print(f"Score: {score}")
        print(f"Param: {param}")
        print(f"Best score: {best_result['score']}")
        print(f"Best param: {best_result['param']}")
        print()
