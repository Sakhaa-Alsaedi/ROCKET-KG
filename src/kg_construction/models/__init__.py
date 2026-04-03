"""ROCKET model registry.

All graph-based models follow the same forward-pass interface:

    logits = model(node_ids, rel_ids, edge_index, batch, visit_node, ehr_nodes)

Use :func:`build_model` as a unified factory instead of importing each class
directly — it instantiates the right model from a string name and a config dict.

Example::

    from src.models import build_model

    model = build_model("CADI", num_nodes=4265, num_rels=1107,
                        max_visit=10, embedding_dim=128, hidden_dim=128,
                        out_channels=2)
"""

from .cadi import CADI
from .cat import CAT
from .baselines.bat import BAT
from .baselines.gnns import GAT, GIN
from .baselines.ehr_baselines import RNN, TransformerBaseline, RETAIN, MLP

__all__ = [
    "CADI", "CAT", "BAT", "GAT", "GIN",
    "RNN", "TransformerBaseline", "RETAIN", "MLP",
    "build_model",
]

# ---------------------------------------------------------------------------
# Unified factory
# ---------------------------------------------------------------------------

_GNN_REGISTRY = {
    "CADI": CADI,
    "CAT": CAT,
    "BAT": BAT,
    "GAT": GAT,
    "GIN": GIN,
}

_EHR_REGISTRY = {
    "RNN": RNN,
    "Transformer": TransformerBaseline,
    "RETAIN": RETAIN,
    "MLP": MLP,
}

_REGISTRY = {**_GNN_REGISTRY, **_EHR_REGISTRY}


def build_model(model_name: str, **kwargs):
    """Instantiate a ROCKET model by name.

    Args:
        model_name: One of ``"CADI"``, ``"CAT"``, ``"BAT"``, ``"GAT"``,
                    ``"GIN"``, ``"RNN"``, ``"Transformer"``, ``"RETAIN"``,
                    ``"MLP"``.
        **kwargs:   Constructor arguments passed directly to the model class.

    Returns:
        Instantiated ``nn.Module``.

    Raises:
        ValueError: If ``model_name`` is not registered.

    Examples::

        model = build_model("CADI", num_nodes=4265, num_rels=1107,
                            max_visit=10, embedding_dim=128, hidden_dim=128,
                            out_channels=2, layers=2)

        rnn = build_model("RNN", input_dim=128, hidden_dim=64, out_channels=2)
    """
    cls = _REGISTRY.get(model_name)
    if cls is None:
        raise ValueError(
            f"Unknown model: {model_name!r}. "
            f"Available: {sorted(_REGISTRY)}"
        )
    return cls(**kwargs)
