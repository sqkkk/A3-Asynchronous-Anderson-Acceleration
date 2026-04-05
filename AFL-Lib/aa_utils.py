import torch


def history_matrix(history, window_size, key='r'):
    recent = history[-min(window_size, len(history)):]
    if len(recent) == 0:
        return None
    return torch.stack([entry[key].reshape(-1) for entry in recent], dim=1)


def recent_residual_matrix(history, window_size):
    # R = [r_{k-m+1}, ..., r_k] stores recent GD residuals / base steps.
    return history_matrix(history, window_size, key='r')


def recent_y_matrix(history, window_size):
    # Y = [y_{k-m+1}, ..., y_k] stores recent base GD iterates.
    return history_matrix(history, window_size, key='y')


def condition_number(matrix, eps=1e-12):
    if matrix is None or matrix.numel() == 0:
        return float('inf')

    singular_values = torch.linalg.svdvals(matrix)
    if singular_values.numel() == 0:
        return float('inf')

    min_sv = singular_values[-1].item()
    max_sv = singular_values[0].item()
    if min_sv <= eps:
        return float('inf')
    return max_sv / min_sv


def solve_anderson_coefficients(residual_matrix, ridge):
    if residual_matrix is None:
        raise ValueError('Residual matrix is empty.')

    device = residual_matrix.device
    dtype = residual_matrix.dtype
    width = residual_matrix.shape[1]

    if width == 0:
        raise ValueError('Residual matrix has zero columns.')

    gram = residual_matrix.T @ residual_matrix
    gram = gram + ridge * torch.eye(width, device=device, dtype=dtype)

    ones = torch.ones((width, 1), device=device, dtype=dtype)
    kkt = torch.zeros((width + 1, width + 1), device=device, dtype=dtype)
    kkt[:width, :width] = gram
    kkt[:width, width:] = ones
    kkt[width:, :width] = ones.T

    rhs = torch.zeros(width + 1, device=device, dtype=dtype)
    rhs[width] = 1.0

    solution = torch.linalg.solve(kkt, rhs)
    return solution[:width]


def combine_iterates(y_matrix, alpha):
    if y_matrix is None:
        raise ValueError('Y matrix is empty.')
    return y_matrix @ alpha


def restart_history(history, keep_last=1):
    if keep_last <= 0:
        history.clear()
    else:
        history[:] = history[-keep_last:]

