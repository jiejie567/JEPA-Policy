"""Numerically frozen spectral metrics and synthetic controls."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


EPS = 1e-12


@dataclass(frozen=True)
class SpectralMetrics:
    n_samples: int
    dimension: int
    maximum_centered_rank: int
    relative_centered_energy: float
    centered_effective_rank: float
    normalized_effective_rank: float
    rankme_raw: float
    ev1: float
    cev8: float
    te8: float
    cev32: float
    te32: float
    normalized_dimension_variance_min: float
    normalized_dimension_variance_median: float
    degenerate: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _validate_embedding(embedding: torch.Tensor) -> torch.Tensor:
    if embedding.dim() != 2:
        raise ValueError(f"Expected an [N,D] matrix, got {tuple(embedding.shape)}")
    if embedding.shape[0] < 2 or embedding.shape[1] < 1:
        raise ValueError(f"Embedding matrix is too small: {tuple(embedding.shape)}")
    if not torch.isfinite(embedding).all():
        raise ValueError("Embedding matrix contains non-finite values")
    return embedding.detach().to(device="cpu", dtype=torch.float64)


def gram_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    """Return ascending squared singular values from a float64 Gram matrix."""

    matrix = _validate_embedding(matrix)
    gram = matrix.T @ matrix
    return torch.linalg.eigvalsh(gram).clamp_min(0.0)


def effective_rank_from_eigenvalues(
    eigenvalues: torch.Tensor, *, eps: float = EPS
) -> float:
    """Compute entropy effective rank from sqrt eigenvalues (singular values)."""

    singular_values = torch.sqrt(eigenvalues.clamp_min(0.0))
    total = singular_values.sum()
    if float(total) <= eps:
        return 0.0
    probabilities = singular_values / total
    positive = probabilities > 0
    entropy = -(probabilities[positive] * probabilities[positive].log()).sum()
    return float(entropy.exp())


def _cumulative_energy(eigenvalues: torch.Tensor, k: int, eps: float) -> float:
    total = eigenvalues.sum()
    if float(total) <= eps:
        return 0.0
    return float(eigenvalues[-min(k, eigenvalues.numel()) :].sum() / total)


def compute_spectral_metrics(
    embedding: torch.Tensor, *, eps: float = EPS
) -> SpectralMetrics:
    """Compute all preregistered point metrics for one representation matrix."""

    raw = _validate_embedding(embedding)
    centered = raw - raw.mean(dim=0, keepdim=True)
    centered_eigenvalues = gram_eigenvalues(centered)
    raw_eigenvalues = gram_eigenvalues(raw)
    centered_energy = centered.square().sum() / raw.shape[0]
    raw_energy = raw.square().sum() / raw.shape[0]
    relative_energy = float(centered_energy / (raw_energy + eps))
    degenerate = float(centered_eigenvalues.sum()) <= eps
    maximum_rank = min(raw.shape[0] - 1, raw.shape[1])

    if degenerate:
        centered_rank = 0.0
        ev1 = cev8 = te8 = cev32 = te32 = 0.0
    else:
        centered_rank = effective_rank_from_eigenvalues(
            centered_eigenvalues, eps=eps
        )
        ev1 = _cumulative_energy(centered_eigenvalues, 1, eps)
        cev8 = _cumulative_energy(centered_eigenvalues, 8, eps)
        cev32 = _cumulative_energy(centered_eigenvalues, 32, eps)
        te8 = max(0.0, 1.0 - cev8)
        te32 = max(0.0, 1.0 - cev32)

    dimension_variance = centered.square().mean(dim=0)
    mean_dimension_energy = raw_energy / raw.shape[1]
    normalized_variance = dimension_variance / (mean_dimension_energy + eps)
    return SpectralMetrics(
        n_samples=raw.shape[0],
        dimension=raw.shape[1],
        maximum_centered_rank=maximum_rank,
        relative_centered_energy=relative_energy,
        centered_effective_rank=centered_rank,
        normalized_effective_rank=(
            centered_rank / maximum_rank if maximum_rank > 0 else 0.0
        ),
        rankme_raw=effective_rank_from_eigenvalues(raw_eigenvalues, eps=eps),
        ev1=ev1,
        cev8=cev8,
        te8=te8,
        cev32=cev32,
        te32=te32,
        normalized_dimension_variance_min=float(normalized_variance.min()),
        normalized_dimension_variance_median=float(normalized_variance.median()),
        degenerate=degenerate,
    )


def compute_centered_spectrum(
    embedding: torch.Tensor, *, eps: float = EPS
) -> dict:
    """Return the complete centered spectrum for archived/representative plots."""

    raw = _validate_embedding(embedding)
    centered = raw - raw.mean(dim=0, keepdim=True)
    eigenvalues = gram_eigenvalues(centered).flip(0)
    maximum_rank = min(raw.shape[0] - 1, raw.shape[1])
    eigenvalues = eigenvalues[:maximum_rank]
    singular_values = torch.sqrt(eigenvalues)
    energy_total = eigenvalues.sum()
    singular_total = singular_values.sum()
    if float(energy_total) <= eps:
        explained_energy = torch.zeros_like(eigenvalues)
        normalized_singular_values = torch.zeros_like(singular_values)
    else:
        explained_energy = eigenvalues / energy_total
        normalized_singular_values = singular_values / singular_total.clamp_min(eps)
    return {
        "maximum_centered_rank": maximum_rank,
        "normalized_singular_values_l1": normalized_singular_values.tolist(),
        "explained_energy": explained_energy.tolist(),
        "cumulative_explained_energy": explained_energy.cumsum(0).tolist(),
        "degenerate": bool(float(energy_total) <= eps),
    }


def compute_control_summary(
    embedding: torch.Tensor,
    *,
    tail_energies: tuple[float, ...] = (0.0, 0.01, 0.05, 0.10),
    eps: float = EPS,
) -> dict:
    """Compute paired low-rank calibration scalars from one eigendecomposition.

    Synthetic tail controls calibrate TE32 only. Their effective ranks are
    descriptive and are not treated as universal rank thresholds.
    """
    raw = _validate_embedding(embedding)
    centered = raw - raw.mean(dim=0, keepdim=True)
    eigenvalues = gram_eigenvalues(centered)
    singular_values = torch.sqrt(eigenvalues).flip(0)
    maximum_rank = min(raw.shape[0] - 1, raw.shape[1])
    if maximum_rank <= 32:
        raise ValueError(
            "Rank-32 controls require min(N - 1, D) greater than 32"
        )
    singular_values = singular_values[:maximum_rank]
    learned_rank = effective_rank_from_eigenvalues(eigenvalues, eps=eps)
    controls = {}
    for rank in (1, 8, 32):
        retained = singular_values[:rank]
        control_rank = effective_rank_from_eigenvalues(
            retained.square(), eps=eps
        )
        controls[f"rank{rank}"] = {
            "algebraic_rank_upper_bound": rank,
            "centered_effective_rank": control_rank,
            "normalized_effective_rank": control_rank / maximum_rank,
            "learned_minus_control_effective_rank": learned_rank - control_rank,
        }

    head = singular_values[:32]
    head_energy = head.square().sum()
    if float(head_energy) <= eps:
        raise ValueError("Cannot calibrate Rank-32 tails with zero head energy")
    tail_dimension = maximum_rank - 32
    tail_controls = {}
    for tail_energy in tail_energies:
        if not 0.0 <= tail_energy < 1.0:
            raise ValueError("tail energies must be in [0,1)")
        calibrated = torch.zeros_like(singular_values)
        calibrated[:32] = head
        if tail_energy > 0:
            tail_total = tail_energy / (1.0 - tail_energy) * head_energy
            calibrated[32:] = torch.sqrt(tail_total / tail_dimension)
        control_rank = effective_rank_from_eigenvalues(
            calibrated.square(), eps=eps
        )
        tail_controls[f"te32_{tail_energy:.2f}"] = {
            "te32": tail_energy,
            "centered_effective_rank": control_rank,
            "normalized_effective_rank": control_rank / maximum_rank,
        }
    return {
        "maximum_centered_rank": maximum_rank,
        "rank_controls": controls,
        "rank32_tail_controls": tail_controls,
        "tail_control_effective_rank_is_descriptive_only": True,
    }


def constant_control(embedding: torch.Tensor) -> torch.Tensor:
    matrix = _validate_embedding(embedding)
    return matrix.mean(dim=0, keepdim=True).expand_as(matrix).clone()


def rank_k_control(embedding: torch.Tensor, k: int) -> torch.Tensor:
    """Return a paired algebraic-rank-at-most-k approximation with its mean."""

    matrix = _validate_embedding(embedding)
    maximum_rank = min(matrix.shape[0] - 1, matrix.shape[1])
    if not 1 <= k <= maximum_rank:
        raise ValueError(f"k must be in [1,{maximum_rank}], got {k}")
    mean = matrix.mean(dim=0, keepdim=True)
    centered = matrix - mean
    u, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    approximation = (u[:, :k] * singular_values[:k]) @ vh[:k]
    return approximation + mean


def rank32_tail_control(
    embedding: torch.Tensor, tail_energy: float
) -> torch.Tensor:
    """Construct a paired control with exactly the requested TE32 calibration."""

    if not 0.0 <= tail_energy < 1.0:
        raise ValueError("tail_energy must be in [0,1)")
    matrix = _validate_embedding(embedding)
    maximum_rank = min(matrix.shape[0] - 1, matrix.shape[1])
    if maximum_rank <= 32:
        raise ValueError(
            "Rank-32 tail controls require min(N - 1, D) greater than 32"
        )
    mean = matrix.mean(dim=0, keepdim=True)
    centered = matrix - mean
    u, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    calibrated = torch.zeros_like(singular_values)
    calibrated[:32] = singular_values[:32]
    head_energy = calibrated[:32].square().sum()
    if float(head_energy) <= EPS:
        raise ValueError("Cannot calibrate a Rank-32 tail with zero head energy")
    available_tail = maximum_rank - 32
    desired_tail_energy = tail_energy / (1.0 - tail_energy) * head_energy
    if tail_energy > 0:
        calibrated[32:maximum_rank] = torch.sqrt(
            desired_tail_energy / available_tail
        )
    control = (u * calibrated) @ vh
    return control + mean


def calibrate_numerical_degeneracy(
    constant_embeddings: list[torch.Tensor], *, eps: float = EPS
) -> float:
    if not constant_embeddings:
        raise ValueError("At least one constant control is required")
    maximum_residual = max(
        compute_spectral_metrics(control, eps=eps).relative_centered_energy
        for control in constant_embeddings
    )
    return max(eps, 100.0 * maximum_residual)
