import torch

from alg.aa_utils import add_aa_common_args, normalize_aa_args, staleness_weight
from alg.asyncaa import Client as AsyncAAClient
from alg.asyncaa import Server as AsyncAAServer
from utils.time_utils import time_record


def add_args(parser):
    add_aa_common_args(parser)
    parser.add_argument('--mu', type=float, default=0.01, help='Proximal regularization strength')
    parser.add_argument('--het_variant', type=str, default='v1', help='v1/v2')
    parser.add_argument('--cv_server_lr', type=float, default=1.0, help='Server control-variate update scale')
    args = parser.parse_args()
    args = normalize_aa_args(args)
    args.het_variant = args.het_variant.lower()
    return args


class Client(AsyncAAClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.mu = args.mu
        self.het_variant = args.het_variant.lower()
        self.C = torch.zeros_like(self.model2tensor())
        self.delta_C = torch.zeros_like(self.model2tensor())

    def _train_corrected_local_problem(self):
        reference_params = [param.detach().clone() for param in self.model.parameters()]
        total_loss = 0.0

        for _ in range(self.epoch):
            for data in self.loader_train:
                X, y = self.preprocess(data)
                preds = self.model(X)
                loss = self.loss_func(preds, y)

                if self.mu > 0:
                    prox_penalty = 0.0
                    for param, ref_param in zip(self.model.parameters(), reference_params):
                        prox_penalty = prox_penalty + torch.sum((param - ref_param) ** 2)
                    loss = loss + 0.5 * self.mu * prox_penalty

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                total_loss += loss.item()

        self.metric['loss'] = total_loss / max(1, len(self.loader_train))

    @time_record
    def run(self):
        model_before = self.model2tensor()
        self._train_corrected_local_problem()

        if self.het_variant == 'v2':
            effective_lr = self.optim.param_groups[0]['lr']
            correction = self.server.C - self.C
            corrected_model = self.model2tensor() - effective_lr * correction
            self.tensor2model(corrected_model)

            denom = max(1, self.local_steps) * max(effective_lr, 1e-12)
            c_next = self.C - self.server.C + (model_before - self.model2tensor()) / denom
            self.delta_C = c_next - self.C
            self.C = c_next.detach().clone()
        else:
            self.delta_C.zero_()

        self.upload_update = self.model2tensor() - model_before

    def comm_bytes(self):
        total_bytes = super().comm_bytes()
        if self.het_variant == 'v2':
            total_bytes += self.C.numel() * self.C.element_size()
        return total_bytes


class Server(AsyncAAServer):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)
        self.het_variant = args.het_variant.lower()
        self.cv_server_lr = args.cv_server_lr
        self.C = torch.zeros_like(self.model2tensor()) if self.het_variant == 'v2' else None

    def aggregate(self):
        super().aggregate()

        if self.het_variant == 'v2':
            stale_scale = staleness_weight(
                self.staleness[self.cur_client.id],
                self.args.aa_stale_strategy,
                self.args.aa_stale_a,
                self.args.aa_stale_b,
            )
            delta_c = torch.nan_to_num(self.cur_client.delta_C.detach().clone())
            self.C = self.C + self.cv_server_lr * stale_scale * delta_c / max(1, self.client_num)
