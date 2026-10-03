"""Pose-delta composition. Translation is in metres; rotation vectors in radians.

Both delta components use the SAME frame. Body deltas right-compose rotation;
world deltas left-compose it. This is not Euler-angle addition or a coupled
SE(3) twist exponential: the translation component is a position increment.
"""
from __future__ import annotations

import torch


def rotation_vector_to_matrix(vector: torch.Tensor) -> torch.Tensor:
    if vector.shape[-1:] != (3,) or not vector.is_floating_point():
        raise ValueError("rotation vector must be a floating tensor [...,3]")
    if not torch.isfinite(vector).all():
        raise ValueError("rotation vector must be finite")
    # sinc form has well-defined limits and gradients at zero rotation.
    angle = torch.linalg.vector_norm(vector, dim=-1)
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1)
    skew = skew.reshape(*vector.shape[:-1], 3, 3)
    a = torch.sinc(angle / torch.pi)[..., None, None]
    b = (0.5 * torch.sinc(angle / (2 * torch.pi)).square())[..., None, None]
    eye = torch.eye(3, dtype=vector.dtype, device=vector.device)
    return eye + a * skew + b * (skew @ skew)


def compose_pose_delta(pose: torch.Tensor, delta: torch.Tensor,
                       frame: str = "body") -> torch.Tensor:
    """Return target [...,4,4] from pose [...,4,4] and delta [...,6].

    The caller must pass the measured current pose each control cycle, then
    solve IK with its own robot model and check reachability before execution.
    """
    if frame not in {"body", "world"}:
        raise ValueError("frame must be body or world")
    if pose.shape[-2:] != (4, 4) or delta.shape != (*pose.shape[:-2], 6):
        raise ValueError("expected pose [...,4,4] and matching delta [...,6]")
    if not pose.is_floating_point() or not delta.is_floating_point():
        raise ValueError("pose and delta must be floating tensors")
    if pose.device != delta.device or pose.dtype != delta.dtype:
        raise ValueError("pose and delta must share dtype and device")
    if not torch.isfinite(pose).all() or not torch.isfinite(delta).all():
        raise ValueError("pose and delta must be finite")
    last_row = pose.new_tensor([0, 0, 0, 1]).expand_as(pose[..., 3, :])
    if not torch.allclose(pose[..., 3, :], last_row, atol=1e-5, rtol=0):
        raise ValueError("pose must be a homogeneous transform")
    rotation = pose[..., :3, :3]
    identity = torch.eye(3, device=pose.device, dtype=pose.dtype).expand_as(rotation)
    if (not torch.allclose(rotation.transpose(-1, -2) @ rotation, identity,
                           atol=1e-4, rtol=1e-4)
            or not torch.allclose(torch.linalg.det(rotation),
                                  torch.ones_like(rotation[..., 0, 0]), atol=1e-4)):
        raise ValueError("pose must contain a proper rotation matrix")
    increment = rotation_vector_to_matrix(delta[..., 3:])
    output = pose.clone()
    if frame == "body":
        output[..., :3, :3] = rotation @ increment
        translation = (rotation @ delta[..., :3, None]).squeeze(-1)
    else:
        output[..., :3, :3] = increment @ rotation
        translation = delta[..., :3]
    output[..., :3, 3] = pose[..., :3, 3] + translation
    return output
