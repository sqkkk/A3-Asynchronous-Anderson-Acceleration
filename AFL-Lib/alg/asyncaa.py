from collections import deque

import torch

from alg.asyncbase import AsyncBaseClient, AsyncBaseServer
from alg.aa_utils import (
    accept_direction,
    add_aa_common_args,
    history_matrix,
    make_stale_aware_update,
    matrix_condition_number,
    mean_direction,
    normalize_aa_args,
    restart_history,
    should_store_update,
    solve_anderson_direction,
)
from utils.time_utils import time_record


def add_args(parser):
    add_aa_common_args(parser)
    args = parser.parse_args()
    return normalize_aa_args(args)


class Client(AsyncBaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.local_steps = max(1, self.epoch * len(self.loader_train))
        self.upload_update = torch.zeros_like(self.model2tensor())

    @time_record
    def run(self):
        model_before = self.model2tensor()
        self.train()
        self.upload_update = self.model2tensor() - model_before


class Server(AsyncBaseServer):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)
        self.eta_g = args.eta_g
        self.history = deque(maxlen=args.aa_memory)

        self.aa_attempts = 0
        self.aa_accepts = 0
        self.aa_restarts = 0
        self.aa_last_condition = 0.0
        self.aa_last_reason = 'warmup'
        self.aa_last_alpha = None

    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()
        self.update_status()

    def _base_direction(self, fallback_update):
        if len(self.history) == 0:
            return fallback_update
        return mean_direction(self.history, self.args.aa_base_window)

    def _should_trigger_aa(self):
        if len(self.history) < self.args.aa_window:
            return False
        return ((self.round + 1) % self.args.aa_trigger_interval) == 0

    def _aa_direction(self, base_direction):
        matrix = history_matrix(self.history, self.args.aa_window)
        cond_num = matrix_condition_number(matrix)
        self.aa_last_condition = cond_num
        if cond_num > self.args.aa_restart_cond:
            restart_history(self.history, self.args.aa_restart_keep)
            self.aa_restarts += 1
            self.aa_last_alpha = None
            return None, 'restart_cond'

        try:
            candidate, alpha = solve_anderson_direction(matrix, self.args.aa_lambda)
        except RuntimeError:
            restart_history(self.history, self.args.aa_restart_keep)
            self.aa_restarts += 1
            self.aa_last_alpha = None
            return None, 'solver_failure'

        accept, reason = accept_direction(candidate, base_direction, self.args)
        if not accept:
            self.aa_last_alpha = alpha.detach().cpu()
            return None, reason

        self.aa_last_alpha = alpha.detach().cpu()
        return candidate, 'accepted'

    def aggregate(self):
        raw_update = torch.nan_to_num(self.cur_client.upload_update.detach().clone())
        staleness = self.staleness[self.cur_client.id]
        normalized_update = make_stale_aware_update(raw_update, staleness, self.cur_client.local_steps, self.args)
        stored = should_store_update(staleness, self.args)

        if stored:
            self.history.append({
                'vector': normalized_update,
                'client_id': self.cur_client.id,
                'staleness': staleness,
                'local_steps': self.cur_client.local_steps,
            })

        direction = self._base_direction(normalized_update) if stored else normalized_update
        self.aa_last_reason = 'base'

        if stored and self._should_trigger_aa():
            self.aa_attempts += 1
            aa_direction, reason = self._aa_direction(direction)
            self.aa_last_reason = reason
            if aa_direction is not None:
                direction = aa_direction
                self.aa_accepts += 1

        self.tensor2model(self.model2tensor() + self.eta_g * direction)
