import math

import torch


def add_aa_common_args(parser):
    parser.add_argument('--eta_g', type=float, default=0.3, help='Global step size for aggregated update deltas')
    parser.add_argument('--aa_window', type=int, default=5, help='AA window size m')
    parser.add_argument('--aa_memory', type=int, default=10, help='Async history pool size M')
    parser.add_argument('--aa_base_window', type=int, default=10, help='Fallback base-direction window size')
    parser.add_argument('--aa_trigger_interval', type=int, default=5, help='Trigger AA every K arrivals')
    parser.add_argument('--aa_lambda', type=float, default=1e-3, help='Ridge regularization in AA subproblem')
    parser.add_argument('--aa_stale_strategy', type=str, default='hinge', help='constant/poly/hinge')
    parser.add_argument('--aa_stale_a', type=float, default=1.0, help='Staleness weighting parameter a')
    parser.add_argument('--aa_stale_b', type=float, default=4.0, help='Staleness weighting parameter b')
    parser.add_argument('--aa_norm', type=str, default='linear', help='none/sqrt/linear')
    parser.add_argument('--aa_max_staleness', type=int, default=-1, help='Negative value disables stale-update filtering')
    parser.add_argument('--aa_restart_cond', type=float, default=1e6, help='Restart AA memory when cond exceeds this value')
    parser.add_argument('--aa_restart_keep', type=int, default=1, help='How many newest vectors to keep after restart')
    parser.add_argument('--aa_accept_min_ratio', type=float, default=0.1, help='Minimum ||d_AA|| / ||d_base|| ratio')
    parser.add_argument('--aa_accept_max_ratio', type=float, default=1.5, help='Maximum ||d_AA|| / ||d_base|| ratio')
    parser.add_argument('--aa_accept_min_cos', type=float, default=0.0, help='Minimum cosine similarity with base direction')
    return parser


def normalize_aa_args(args):
    args.aa_window = max(1, args.aa_window)
    args.aa_memory = max(args.aa_memory, args.aa_window, 1)
    args.aa_base_window = max(1, min(args.aa_base_window, args.aa_memory))
    args.aa_trigger_interval = max(1, args.aa_trigger_interval)
    args.aa_restart_keep = max(0, min(args.aa_restart_keep, args.aa_memory))
    return args


def staleness_weight(staleness, strategy, a, b):
    tau = max(int(staleness), 0)
    if strategy == 'poly':
        return 1.0 / ((tau + 1.0) ** max(abs(a), 1e-12))
    if strategy == 'hinge':
        if tau <= b:
            return 1.0
        return 1.0 / (abs(a) * (tau - b) + 1.0)
    return 1.0


def work_normalizer(local_steps, mode):
    steps = max(int(local_steps), 1)
    if mode == 'sqrt':
        return math.sqrt(steps)
    if mode == 'linear':
        return float(steps)
    return 1.0


def make_stale_aware_update(update, staleness, local_steps, args):
    scale = staleness_weight(staleness, args.aa_stale_strategy, args.aa_stale_a, args.aa_stale_b)
    scale /= work_normalizer(local_steps, args.aa_norm)
    return update.detach().clone() * scale


def should_store_update(staleness, args):
    return args.aa_max_staleness < 0 or staleness <= args.aa_max_staleness


def recent_history(history, window_size):
    return list(history)[-min(len(history), window_size):]


def mean_direction(history, window_size):
    vectors = [entry['vector'] for entry in recent_history(history, window_size)]
    if not vectors:
        raise ValueError('Cannot build a base direction from an empty history.')
    return torch.mean(torch.stack(vectors, dim=0), dim=0)


def history_matrix(history, window_size):
    vectors = [entry['vector'] for entry in recent_history(history, window_size)]
    if not vectors:
        raise ValueError('Cannot build an AA matrix from an empty history.')
    return torch.stack(vectors, dim=1)


def matrix_condition_number(matrix, eps=1e-12):
    singular_values = torch.linalg.svdvals(matrix.double())
    if singular_values.numel() == 0:
        return float('inf')
    min_sv = singular_values[-1].item()
    if min_sv <= eps:
        return float('inf')
    return singular_values[0].item() / min_sv


def solve_anderson_direction(matrix, ridge):
    gram = torch.matmul(matrix.t(), matrix).double()
    if ridge > 0:
        gram = gram + ridge * torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)

    ones = torch.ones(gram.size(0), device=gram.device, dtype=gram.dtype)
    try:
        solve_ones = torch.linalg.solve(gram, ones)
    except RuntimeError:
        solve_ones = torch.matmul(torch.linalg.pinv(gram), ones)

    denom = torch.dot(ones, solve_ones)
    if torch.abs(denom) < 1e-12:
        raise RuntimeError('AA affine constraint is ill-conditioned.')

    alpha = solve_ones / denom
    alpha = alpha.to(matrix.dtype)
    direction = torch.matmul(matrix, alpha)
    return direction, alpha


def cosine_similarity(vec_a, vec_b, eps=1e-12):
    norm_a = torch.norm(vec_a)
    norm_b = torch.norm(vec_b)
    if norm_a.item() <= eps or norm_b.item() <= eps:
        return 1.0
    return torch.dot(vec_a, vec_b).item() / (norm_a.item() * norm_b.item() + eps)


def accept_direction(candidate, base_direction, args):
    if candidate is None or not torch.isfinite(candidate).all():
        return False, 'non_finite'

    base_norm = torch.norm(base_direction).item()
    cand_norm = torch.norm(candidate).item()
    if cand_norm <= 1e-12:
        return False, 'tiny_candidate'
    if base_norm <= 1e-12:
        return True, 'accepted'

    ratio = cand_norm / (base_norm + 1e-12)
    if ratio < args.aa_accept_min_ratio:
        return False, 'ratio_too_small'
    if ratio > args.aa_accept_max_ratio:
        return False, 'ratio_too_large'

    cos = cosine_similarity(candidate, base_direction)
    if cos < args.aa_accept_min_cos:
        return False, 'negative_alignment'
    return True, 'accepted'


def restart_history(history, keep_last):
    kept = list(history)[-keep_last:] if keep_last > 0 else []
    history.clear()
    history.extend(kept)
