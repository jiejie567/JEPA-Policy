"""Fail-closed infrastructure for the Future Rollout Audit."""

__all__ = [
    "FutureSpaceContractError",
    "JointSamplerResult",
    "JointSamplerTrace",
    "build_future_space_contract",
    "encode_audit_future_target",
    "validate_future_space_contract",
]


def __getattr__(name):
    """Keep spawn-based environment workers free of model/Torch imports."""

    if name in {"JointSamplerResult", "JointSamplerTrace"}:
        from mip import samplers

        return getattr(samplers, name)
    if name in {
        "FutureSpaceContractError",
        "build_future_space_contract",
        "encode_audit_future_target",
        "validate_future_space_contract",
    }:
        from mip.future_rollout_audit import contracts

        return getattr(contracts, name)
    raise AttributeError(name)
