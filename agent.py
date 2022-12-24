import math
import time

import torch
from typing import Tuple
from src import FEATURE_DIM, RADIUS, splev, N_CTPS, P
from net import NeuralNetwork

class Agent:

    def __init__(self) -> None:
        """Initialize the agent, e.g., load the classifier model. """

        self.model = NeuralNetwork()
        self.model.load_state_dict(torch.load("model_weights.pth"))
        self.model.eval()

        self.t = torch.linspace(0, N_CTPS - P, 100)
        self.knots = torch.cat([
            torch.zeros(P),
            torch.arange(N_CTPS + 1 - P),
            torch.full((P,), N_CTPS - P),
        ])
        self.ctp_head = torch.tensor([[0., 0.]])
        self.ctp_end = torch.tensor([[N_CTPS, 0.]])

    def evaluate(
        self,
        traj: torch.Tensor,
        target_pos: torch.Tensor,
        target_scores: torch.Tensor,
        radius: float = RADIUS,
    ) -> torch.Tensor:

        cdist = torch.cdist(target_pos, traj)
        d = cdist.min(-1).values
        hit = (d < radius)
        value = torch.sum(hit * target_scores, dim=-1)
        return value

    def evaluate_d(
        self,
        traj: torch.Tensor,
        target_pos: torch.Tensor,
        target_scores: torch.Tensor,
        radius: float = RADIUS,
    ) -> torch.Tensor:
        # cdist = torch.cdist(target_pos, traj)
        # d = cdist.min(-1).values

        cd = torch.cdist(traj, target_pos)
        d = cd.min(dim=0).values

        hits_1 = (d < radius)
        hits_m = (d < (radius))
        posit = target_scores > 0

        mask1 = posit & hits_1
        mask2 = posit & ~hits_1
        mask3 = ~posit & hits_m
        mask4 = ~posit & ~hits_m

        d[mask1] = d[mask1] * (-0.0001) + 1
        d[mask2] = radius / d[mask2]
        d[mask3] = radius / d[mask3]
        d[mask4] = d[mask4] * (-0.0001) + 1
        value = torch.sum(d * target_scores)
        return value

    def compute_traj(
        self,
        ctps_inter: torch.Tensor
    ) -> torch.Tensor:
        return splev(self.t, self.knots, torch.cat([self.ctp_head, ctps_inter, self.ctp_end]), P)

    def score(
        self,
        ctps_inter: torch.Tensor,
        target_pos: torch.Tensor,
        target_score: torch.Tensor,
        radius: float = RADIUS
    ) -> torch.Tensor:

        return self.evaluate(self.compute_traj(ctps_inter), target_pos, target_score, radius)

    def get_action(self,
        target_pos: torch.Tensor,
        target_features: torch.Tensor,
        class_scores: torch.Tensor,
        stdx: float = N_CTPS - 2,
        stdy: float = 2.,
        meanx: float = 2.5,
        meany: float = 0.
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        assert len(target_pos) == len(target_features)

        t_start = time.time()

        y_pred = self.model(target_features)
        y_pred_class = torch.argmax(y_pred, dim=1)
        target_score = class_scores[y_pred_class]

        ctps_inter = torch.zeros(N_CTPS - 2, 2)
        ctps_inter[:, 0] = torch.linspace(1, N_CTPS - 1, N_CTPS - 2)

        best_inter = None
        best_score = -100000

        for i in range(150):

            ctps_inter = torch.normal(0, 1, size=(N_CTPS - 2, 2)) * torch.tensor([stdx, stdy]) + torch.tensor([meanx, meany])
            score = self.score(ctps_inter, target_pos, target_score)
            if score >= best_score:
                best_score = score
                best_inter = torch.clone(ctps_inter)

        # print(f"Best score using random: {best_score}")


        # t_start = time.time()
        # #
        # n = 1
        # ctps_inters = [torch.zeros(N_CTPS - 2, 2) for _ in range(n)]
        # #
        # for ctps_inter in ctps_inters:
        #     torch.randn
        #     ctps_inter.data = torch.rand((N_CTPS - 2, 2)) * torch.tensor([N_CTPS - 2, 2.]) + torch.tensor([1., -1.])
        #     ctps_inter.data = best_inter
        #     ctps_inter.requires_grad = True
        #
        # epochs = 20
        #
        # for ctps_inter in ctps_inters:
        #     lr = 0.01
        #
        #     for i in range(epochs):
        #         traj = self.compute_traj(ctps_inter)
        #         score = self.evaluate_d(traj, target_pos, target_score)
        #         score.backward()
        #         ctps_inter.data = ctps_inter + lr * ctps_inter.grad / torch.norm(ctps_inter.grad)
        #         print(score)
        #         if i % 10 == 0:
        #             lr /= 2
        #
        #
        #     print()
        #
        # best_inter = ctps_inters[0]
        # for ctps_inter in ctps_inters[1:]:
        #     if self.score(ctps_inter, target_pos, target_score) > self.score(best_inter, target_pos, target_score):
        #         best_inter = ctps_inter
        #
        # print(f"Best score using backwards: {self.score(best_inter, target_pos, target_score)}")
        # print(f"Time usage: {time.time() - t_start: .3f}s")

        return best_inter.detach()

