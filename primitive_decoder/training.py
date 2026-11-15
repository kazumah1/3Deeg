"""Training, rendered loss, and conditioning controls for cuboid decoders."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .geometry import matrix_from_rotation_6d
from .matching import matched_parameter_loss


def render_cuboids_torch(tokens, samples_per_component: int = 128):
    """Differentiably render a fixed, deterministic surface design per cuboid."""
    import torch
    batch, components, _ = tokens.shape
    device, dtype = tokens.device, tokens.dtype
    index = torch.arange(samples_per_component, device=device)
    faces = index.remainder(6)
    # Low-discrepancy, deterministic face coordinates make the loss reproducible.
    u = ((index * 0.61803398875) % 1).to(dtype) * 2 - 1
    v = ((index * 0.41421356237) % 1).to(dtype) * 2 - 1
    base = torch.stack((u, v, torch.zeros_like(u)), -1)[None, None].expand(batch, components, -1, -1).clone()
    half = tokens[..., 3:6].exp()
    for axis in range(3):
        other = [item for item in range(3) if item != axis]
        mask_negative, mask_positive = faces == 2 * axis, faces == 2 * axis + 1
        base[..., axis][..., mask_negative] = -1
        base[..., axis][..., mask_positive] = 1
        base[..., other[0]][..., mask_negative | mask_positive] = u[mask_negative | mask_positive]
        base[..., other[1]][..., mask_negative | mask_positive] = v[mask_negative | mask_positive]
    local = base * half[:, :, None, :]
    rotation = matrix_from_rotation_6d(tokens[..., 6:])
    world = tokens[..., None, :3] + local @ rotation.transpose(-1, -2)
    return world.reshape(batch, components * samples_per_component, 3)


def chamfer_loss_torch(prediction, target):
    distances = torch.cdist(prediction, target).square()
    return distances.amin(-1).mean() + distances.amin(-2).mean()


def primitive_objective(predicted, target_tokens, target_points, parameter_weight: float = 1.0, rendered_weight: float = 1.0):
    parameter = matched_parameter_loss(predicted, target_tokens)["total"]
    rendered = chamfer_loss_torch(render_cuboids_torch(predicted), target_points)
    return parameter_weight * parameter + rendered_weight * rendered, {"parameter": parameter.detach(), "rendered": rendered.detach()}


@dataclass
class TrainingResult:
    state_dict: dict
    history: list[dict]


def category_template_baseline(tokens, categories, query_categories):
    """Build category mean sets after Hungarian-aligning each set to an anchor."""
    import torch
    from .matching import hungarian_indices
    templates = {}
    for category in sorted(set(map(str, categories))):
        members = tokens[np.asarray(categories) == category]
        anchor = members[0]
        aligned = [member[torch.as_tensor(hungarian_indices(anchor, member), device=member.device)] for member in members]
        templates[category] = torch.stack(aligned).mean(0)
    return torch.stack([templates[str(category)] for category in query_categories])


def fit_decoder(model, train_loader, validation_loader, *, steps: int, seed: int, lr: float = 1e-3, device: str = "cuda") -> TrainingResult:
    """Fit a direct primitive decoder using the same seed/budget for each baseline."""
    import torch
    torch.manual_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    history, iterator = [], iter(train_loader)
    best_state, best_value = None, float("inf")
    for step in range(1, steps + 1):
        try: batch = next(iterator)
        except StopIteration: iterator = iter(train_loader); batch = next(iterator)
        embedding, tokens, points = (batch[key].to(device) for key in ("fg", "tokens", "points"))
        optimizer.zero_grad(); loss, values = primitive_objective(model(embedding), tokens, points); loss.backward(); optimizer.step()
        if step == 1 or step % max(1, steps // 20) == 0:
            model.eval(); validation = []
            with torch.no_grad():
                for held in validation_loader:
                    held = {key: value.to(device) for key, value in held.items()}
                    validation.append(float(primitive_objective(model(held["fg"]), held["tokens"], held["points"])[0]))
            value = float(np.mean(validation)); history.append({"step": step, "train_loss": float(loss), "validation_loss": value,
                                                                "parameter_loss": float(values["parameter"]), "rendered_loss": float(values["rendered"])})
            if value < best_value:
                best_value, best_state = value, {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            model.train()
    return TrainingResult(best_state, history)


def conditioning_controls(model, fg, *, seed: int = 0):
    """Normal, zeroed, and across-object shuffled f_g predictions."""
    import torch
    generator = torch.Generator(device=fg.device).manual_seed(seed)
    return {"normal": model(fg), "zero": model(torch.zeros_like(fg)),
            "shuffled": model(fg[torch.randperm(len(fg), generator=generator, device=fg.device)])}
