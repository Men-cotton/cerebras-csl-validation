"""Numerical CFD-demo oracle; seeded weights are not a trained fluid model.

Contract: DeepMind's meshgraphnets at 34b2486902f082ad3a58bffebdc499c84c93d31c,
core_model.py:35-72,90-120; cfd_model.py:29-61,88-93; cfd_eval.py:23-42.
Input normalizers are explicitly identity. Output normalization uses fixed
mean zero and std 0.01 to exercise physical updates with random weights.
Neither these statistics nor the weights reproduce the paper's accuracy.

Dense NumPy matrix products provide the default oracle. The optional factored
edge path independently exercises first-layer node projection reuse; floating
point addition order differs, so equality is approximate.
"""

from __future__ import annotations

from typing import Any

import numpy as np

LN_EPSILON = np.float32(1e-5)
OUTPUT_STD = np.float32(0.01)
OUTPUT_MEAN = np.float32(0.0)
DEFAULT_SEED = 20261021
NODE_INPUT_SIZE = 11
EDGE_INPUT_SIZE = 3
NODE_TYPE_COUNT = 9


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def weight_layout(H: int, S: int) -> list[dict[str, Any]]:
    """Describe the flat f32 ABI, with absolute offsets in float32 elements.

    MLP order: node_encoder, edge_encoder, processor_b_edge/node for b in
    range(S), decoder. Field order: w1,b1,w2,b2,w3,b3,[gamma,beta]. All matrices
    use C order [input,output]. Shapes are tuples; offsets count f32, not bytes.
    """
    H = _positive_integer(H, "H")
    S = _positive_integer(S, "S")
    specs = [("node_encoder", NODE_INPUT_SIZE, H, True),
             ("edge_encoder", EDGE_INPUT_SIZE, H, True)]
    for block in range(S):
        specs.extend([(f"processor_{block}_edge", 3 * H, H, True),
                      (f"processor_{block}_node", 2 * H, H, True)])
    specs.append(("decoder", H, 2, False))
    offset = 0
    result = []
    for name, n_in, n_out, layer_norm in specs:
        start = offset
        fields = {}
        shapes = [("w1", (n_in, H)), ("b1", (H,)),
                  ("w2", (H, H)), ("b2", (H,)),
                  ("w3", (H, n_out)), ("b3", (n_out,))]
        if layer_norm:
            shapes.extend([("gamma", (n_out,)), ("beta", (n_out,))])
        for field, shape in shapes:
            size = int(np.prod(shape))
            fields[field] = {"offset": offset, "shape": shape, "size": size}
            offset += size
        result.append({"name": name, "input_size": n_in, "output_size": n_out,
                       "layer_norm": layer_norm, "offset": start,
                       "size": offset - start, "fields": fields})
    return result


def weight_count(H: int, S: int) -> int:
    """Number of f32 scalars, including affine biases and LN scale/offset."""
    H = _positive_integer(H, "H")
    S = _positive_integer(S, "S")
    return (6 + 9 * S) * H * H + (28 + 10 * S) * H + 2


def make_weights(H: int, S: int, seed: int = DEFAULT_SEED) -> np.ndarray:
    """Return deterministic, nonuniform, untied numerical-probe weights."""
    layout = weight_layout(H, S)
    rng = np.random.default_rng(seed)
    weights = np.empty(weight_count(H, S), dtype=np.float32)
    for item in layout:
        for name, field in item["fields"].items():
            size = field["size"]
            if name.startswith("w"):
                scale = np.float32(0.6 / np.sqrt(field["shape"][0]))
                values = rng.standard_normal(size).astype(np.float32) * scale
            elif name == "gamma":
                values = (np.float32(0.7) +
                          np.float32(0.6) * rng.random(size).astype(np.float32))
            else:
                values = (rng.standard_normal(size).astype(np.float32) *
                          np.float32(0.03))
            start = field["offset"]
            weights[start:start + size] = values
    return weights


def unpack_weights(weights: np.ndarray, H: int, S: int) -> dict[str, dict[str, np.ndarray]]:
    """Return named views into a validated flat array; no caller data is edited."""
    flat = np.asarray(weights, dtype=np.float32)
    if flat.ndim != 1 or flat.size != weight_count(H, S):
        raise ValueError("weights must have the exact flat ABI length")
    if not np.isfinite(flat).all():
        raise ValueError("weights must be finite")
    result = {}
    for item in weight_layout(H, S):
        fields = {}
        for name, field in item["fields"].items():
            start = field["offset"]
            fields[name] = flat[start:start + field["size"]].reshape(field["shape"])
        result[item["name"]] = fields
    return result


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Population variance over features, with epsilon inside the square root."""
    mean = np.mean(x, axis=-1, keepdims=True, dtype=np.float32)
    centered = x - mean
    variance = np.mean(centered * centered, axis=-1, keepdims=True, dtype=np.float32)
    return ((centered / np.sqrt(variance + LN_EPSILON)) * gamma + beta).astype(np.float32)


def _mlp_tail(first: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    x = np.maximum(first, np.float32(0.0))
    x = np.maximum(x @ params["w2"] + params["b2"], np.float32(0.0))
    x = x @ params["w3"] + params["b3"]
    if "gamma" in params:
        x = layer_norm(x, params["gamma"], params["beta"])
    return x.astype(np.float32)


def apply_mlp(x: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    """Three dense affine layers; only the first two have ReLU."""
    x = np.asarray(x, dtype=np.float32)
    return _mlp_tail(x @ params["w1"] + params["b1"], params)


def process_block(h: np.ndarray, e: np.ndarray, senders: np.ndarray,
                  receivers: np.ndarray, edge_params: dict[str, np.ndarray],
                  node_params: dict[str, np.ndarray], *,
                  factored_edges: bool = False) -> dict[str, np.ndarray]:
    """Apply one block; aggregate edge deltas before either residual addition."""
    H = h.shape[1]
    if factored_edges:
        # Every node projection is computed once, not once per incident edge.
        w = edge_params["w1"]
        projected_sender = h @ w[:H]
        projected_receiver = h @ w[H:2 * H]
        first = (projected_sender[senders] + projected_receiver[receivers] +
                 e @ w[2 * H:] + edge_params["b1"])
        delta_e = _mlp_tail(first, edge_params)
    else:
        edge_input = np.concatenate((h[senders], h[receivers], e), axis=1)
        delta_e = apply_mlp(edge_input, edge_params)
    aggregate = np.zeros_like(h, dtype=np.float32)
    # add.at preserves duplicate edges and receiver ordering, unlike a fancy
    # indexed += operation. Empty receivers retain the additive identity.
    np.add.at(aggregate, receivers, delta_e)
    delta_h = apply_mlp(np.concatenate((h, aggregate), axis=1), node_params)
    return {"h": h + delta_h, "e": e + delta_e, "aggregate": aggregate,
            "delta_e": delta_e, "delta_h": delta_h}


def _inputs(velocity: np.ndarray, node_type: np.ndarray, edge_features: np.ndarray,
            senders: np.ndarray, receivers: np.ndarray) -> tuple[np.ndarray, ...]:
    velocity = np.array(velocity, dtype=np.float32, copy=True)
    if velocity.ndim != 2 or velocity.shape[1] != 2 or velocity.shape[0] == 0:
        raise ValueError("velocity must have shape [N,2] with N>0")
    types = np.asarray(node_type)
    if types.shape == (velocity.shape[0], 1):
        types = types[:, 0]
    if types.shape != (velocity.shape[0],) or types.dtype.kind not in "iu":
        raise ValueError("node_type must contain N integer node types")
    if np.any(types < 0) or np.any(types >= NODE_TYPE_COUNT):
        raise ValueError("node_type must be in [0,9)")
    edge_features = np.asarray(edge_features, dtype=np.float32)
    if edge_features.ndim != 2 or edge_features.shape[1] != 3:
        raise ValueError("edge_features must have shape [E,3]")
    indices = []
    for name, values in (("senders", senders), ("receivers", receivers)):
        values = np.asarray(values)
        if values.shape != (edge_features.shape[0],) or values.dtype.kind not in "iu":
            raise ValueError(f"{name} must contain E integer indices")
        if np.any(values < 0) or np.any(values >= velocity.shape[0]):
            raise ValueError(f"{name} index outside [0,N)")
        indices.append(values.astype(np.intp, copy=False))
    if not np.isfinite(velocity).all() or not np.isfinite(edge_features).all():
        raise ValueError("input features must be finite")
    return velocity, types.astype(np.intp, copy=False), edge_features, *indices


def rollout(velocity: np.ndarray, node_type: np.ndarray, edge_features: np.ndarray,
            senders: np.ndarray, receivers: np.ndarray, weights: np.ndarray,
            H: int, S: int, steps: int, *, output_std: float = OUTPUT_STD,
            output_mean: float = OUTPUT_MEAN, record_history: bool = False,
            factored_edges: bool = False) -> dict[str, Any]:
    """Run physical steps, resetting edge latents from cached geometry each time.

    Returns final velocity, h, e, aggregate, and cached_e0. With record_history,
    history contains each step's input velocity, encoded node latent, all block
    results, decoder output, and updated velocity (copied snapshots). By default
    no per-block history is retained, avoiding O(steps*S*(N+E)*H) storage.
    Boundary velocities (types other than 0/5) are exactly preserved.
    """
    H = _positive_integer(H, "H")
    S = _positive_integer(S, "S")
    steps = _positive_integer(steps, "steps")
    current, types, edge_features, senders, receivers = _inputs(
        velocity, node_type, edge_features, senders, receivers)
    output_std, output_mean = np.float32(output_std), np.float32(output_mean)
    if not np.isfinite(output_std) or output_std <= 0 or not np.isfinite(output_mean):
        raise ValueError("output_std must be finite positive and output_mean finite")
    params = unpack_weights(weights, H, S)
    one_hot = np.eye(NODE_TYPE_COUNT, dtype=np.float32)[types]
    mask = (types == 0) | (types == 5)
    cached_e0 = apply_mlp(edge_features, params["edge_encoder"])
    history = []
    for _ in range(steps):
        h = apply_mlp(np.concatenate((current, one_hot), axis=1), params["node_encoder"])
        e = cached_e0.copy()
        if record_history:
            snapshot = {"input_velocity": current.copy(), "encoded_h": h.copy(),
                        "blocks": []}
        for block in range(S):
            result = process_block(h, e, senders, receivers,
                                   params[f"processor_{block}_edge"],
                                   params[f"processor_{block}_node"],
                                   factored_edges=factored_edges)
            h, e, aggregate = result["h"], result["e"], result["aggregate"]
            if record_history:
                snapshot["blocks"].append({key: value.copy() for key, value in result.items()})
        decoder_output = apply_mlp(h, params["decoder"])
        update = decoder_output * output_std + output_mean
        current[mask] = current[mask] + update[mask]
        if record_history:
            snapshot["decoder_output"] = decoder_output.copy()
            snapshot["velocity"] = current.copy()
            history.append(snapshot)
    return {"velocity": current, "h": h, "e": e, "aggregate": aggregate,
            "cached_e0": cached_e0, "history": history,
            "normalization": {"inputs": "identity_numerical_probe",
                              "output_std": float(output_std),
                              "output_mean": float(output_mean)}}
