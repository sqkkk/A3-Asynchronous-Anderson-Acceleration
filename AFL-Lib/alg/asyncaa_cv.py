import torch

from aa_utils import (
    combine_iterates,
    condition_number,
    recent_residual_matrix,
    recent_y_matrix,
    restart_history,
    solve_anderson_coefficients,
)
from alg.asyncbase import AsyncBaseClient, AsyncBaseServer, Status
from utils.time_utils import time_record


def add_args(parser):
    parser.add_argument('--a', type=int, default=1)
    parser.add_argument('--b', type=int, default=4)
    parser.add_argument('--strategy', type=str, default='hinge', help='constant/poly/hinge')
    parser.add_argument('--aa_memory', type=int, default=10)
    parser.add_argument(
        '--aa_window',
        type=int,
        default=5,
        help='minimum number of received updates before triggering one buffered AA aggregation',
    )
    parser.add_argument('--aa_lambda', type=float, default=1e-4)
    parser.add_argument('--aa_beta', type=float, default=1.0)
    parser.add_argument('--aa_max_condition', type=float, default=1e8)
    parser.add_argument('--aa_residual_tol', type=float, default=5.0)
    parser.add_argument('--eta_g', type=float, default=1e-3)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--adam_eps', type=float, default=1e-8)
    return parser.parse_args()


class Client(AsyncBaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.C = torch.zeros_like(self.model2tensor())

    @time_record
    def run(self):
        self.prev_model = self.model2tensor()

        self.train()
        # Control-variate correction is the dominant drift-control ingredient in FedAC.
        self.tensor2model(self.model2tensor() - self.lr * (self.server.C - self.C))

        cur_model = self.model2tensor()
        hat_C = (self.prev_model - cur_model) / (self.epoch * self.lr) - (self.server.C - self.C)
        self.dC = hat_C - self.C
        self.C = hat_C

        self.dW = cur_model - self.prev_model

    def comm_bytes(self):
        model_tensor = self.model2tensor()
        return model_tensor.numel() * model_tensor.element_size() + self.C.numel() + self.C.element_size()


class Server(AsyncBaseServer):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)
        model_tensor = self.model2tensor()

        self.aa_buffer = []
        self.applied_update = False

        # Control variate + Adam-like server state.
        self.C = torch.zeros_like(model_tensor)
        self.m = torch.zeros_like(model_tensor)
        self.v = torch.zeros_like(model_tensor)
        self.adam_step = 0

    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()
        self.update_status()

    def _staleness_weight(self, tau):
        a = self.args.a
        b = self.args.b
        strategy = self.args.strategy
        if strategy == 'poly':
            return 1 / ((tau + 1) ** abs(a))
        if strategy == 'hinge':
            return 1 / (a * (tau + b) + 1) if tau > b else 1
        return 1

    def _buffer_update(self):
        self.aa_buffer.append({
            'dW': self.cur_client.dW.clone().detach(),
            'dC': self.cur_client.dC.clone().detach(),
            'client_id': self.cur_client.id,
            'staleness': self.staleness[self.cur_client.id],
            'local_steps': self.cur_client.epoch,
        })

        if len(self.aa_buffer) > self.args.aa_memory:
            self.aa_buffer = self.aa_buffer[-self.args.aa_memory:]

    def _current_period_records(self):
        if len(self.aa_buffer) == 0:
            return [], None

        current_entries = list(self.aa_buffer)

        # x_k is the current global iterate. Every buffered control-variate
        # update defines a base Adam/GD iterate y_i from this same x_k.
        x_k = self.model2tensor()
        m_t = self.m.clone().detach()
        v_t = self.v.clone().detach()
        step_t = self.adam_step

        records = []
        for entry in current_entries:
            stale_weight = self._staleness_weight(entry['staleness'])

            # dW is already client-drift-corrected by the control variate.
            d_base = stale_weight * entry['dW']
            g_i = -d_base

            step_t += 1
            m_t = self.args.beta1 * m_t + (1.0 - self.args.beta1) * g_i
            v_t = self.args.beta2 * v_t + (1.0 - self.args.beta2) * (g_i ** 2)

            bias_correction1 = 1.0 - (self.args.beta1 ** step_t)
            bias_correction2 = 1.0 - (self.args.beta2 ** step_t)
            m_hat = m_t / bias_correction1
            v_hat = v_t / bias_correction2

            # y_i = x_k - eta_g * preconditioned(g_i) is the base GD iterate.
            # r_i = y_i - x_k is the GD residual / step stored in R.
            r_i = -self.args.eta_g * m_hat / (torch.sqrt(v_hat) + self.args.adam_eps)
            y_i = x_k + r_i

            records.append({
                'x': x_k.clone().detach(),
                'y': y_i.clone().detach(),
                'r': r_i.clone().detach(),
                'client_id': entry['client_id'],
                'staleness': entry['staleness'],
                'local_steps': entry['local_steps'],
            })

        return records, {
            'm': m_t.clone().detach(),
            'v': v_t.clone().detach(),
            'step': step_t,
        }

    def _update_control_variate(self, current_entries):
        if len(current_entries) == 0:
            return

        weights = [self._staleness_weight(entry['staleness']) for entry in current_entries]
        total_weight = sum(weights)
        if total_weight <= 0:
            weights = [1.0 / len(current_entries) for _ in current_entries]
        else:
            weights = [w / total_weight for w in weights]

        self.C += sum([w * entry['dC'] for w, entry in zip(weights, current_entries)])

    def _base_iterate(self, period_records):
        return period_records[-1]['y']

    def _aa_iterate(self, period_records):
        window_size = len(period_records)
        if window_size < 2:
            return None, {'reason': 'insufficient_history'}

        # R is the residual history matrix built from current-period base steps.
        residual_matrix = recent_residual_matrix(period_records, window_size)
        cond = condition_number(residual_matrix)
        if (not torch.isfinite(torch.tensor(cond))) or cond > self.args.aa_max_condition:
            restart_history(self.aa_buffer, keep_last=0)
            return None, {'reason': 'bad_conditioning', 'condition_number': cond}

        try:
            # alpha solves min ||R alpha||_2^2 + lambda ||alpha||_2^2 subject to sum(alpha)=1.
            alpha = solve_anderson_coefficients(residual_matrix, self.args.aa_lambda)
        except RuntimeError as exc:
            return None, {'reason': 'solver_failure', 'error': str(exc)}

        y_matrix = recent_y_matrix(period_records, window_size)
        blended_alpha = alpha.clone()
        blended_alpha *= self.args.aa_beta
        blended_alpha[-1] += 1.0 - self.args.aa_beta

        # x_next is an affine combination of past/current base GD iterates y_i.
        x_candidate = combine_iterates(y_matrix, blended_alpha)
        if not torch.isfinite(x_candidate).all():
            return None, {'reason': 'non_finite_candidate'}

        mixed_residual = residual_matrix @ blended_alpha
        base_residual_norm = torch.norm(period_records[-1]['r']).item()
        candidate_residual_norm = torch.norm(mixed_residual).item()
        if candidate_residual_norm > self.args.aa_residual_tol * max(base_residual_norm, 1e-12):
            return None, {
                'reason': 'residual_rejection',
                'candidate_residual_norm': candidate_residual_norm,
                'latest_residual_norm': base_residual_norm,
            }

        return x_candidate, {
            'reason': 'accepted',
            'condition_number': cond,
            'candidate_residual_norm': candidate_residual_norm,
            'latest_residual_norm': base_residual_norm,
        }

    def aggregate(self):
        self.applied_update = False
        self._buffer_update()

        flush_size = max(1, min(self.args.aa_window, self.args.aa_memory))
        if len(self.aa_buffer) < flush_size:
            self.metric['aa_status'] = 'buffering'
            return

        current_entries = list(self.aa_buffer)
        period_records, optimizer_state = self._current_period_records()
        y_base = self._base_iterate(period_records)
        x_candidate, aa_info = self._aa_iterate(period_records)

        if x_candidate is None:
            self.tensor2model(y_base)
            self.metric['aa_status'] = aa_info['reason']
        else:
            self.tensor2model(x_candidate)
            self.metric['aa_status'] = aa_info['reason']

        self.m = optimizer_state['m']
        self.v = optimizer_state['v']
        self.adam_step = optimizer_state['step']
        self._update_control_variate(current_entries)

        self.applied_update = True
        self.aa_buffer.clear()

    def update_status(self):
        self.cur_client.status = Status.IDLE

        if self.applied_update:
            for c in filter(lambda x: x.status == Status.ACTIVE, self.clients):
                self.staleness[c.id] += 1
