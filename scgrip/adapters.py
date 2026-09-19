from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from .config import AdapterConfig


AdapterKind = Literal["full_w0", "affine_w0"]


def _parent_and_name(root: nn.Module, path: str) -> tuple[nn.Module, str]:  # принимает путь к модулю 
    pieces = path.split(".")
    if not pieces or any(not piece for piece in pieces):
        raise ValueError(f"Invalid module path: {path!r}")
    parent: nn.Module = root
    for piece in pieces[:-1]:
        try:
            parent = parent.get_submodule(piece)
        except AttributeError as error:
            raise ValueError(f"Cannot resolve {path!r}: missing {piece!r}.") from error
    return parent, pieces[-1]    # и возвращает родительский модуль и имя искомого подмодуля


def _get_child(parent: nn.Module, name: str) -> nn.Module:  # получить подмодуль
    if name in parent._modules:
        return parent._modules[name]
    raise ValueError(f"{type(parent).__name__} has no child module {name!r}.")


def _set_child(parent: nn.Module, name: str, value: nn.Module) -> None:  # заменить подмодуль
    if name not in parent._modules:
        raise ValueError(f"{type(parent).__name__} has no child module {name!r}.")
    parent._modules[name] = value


def infer_expression_layer_path(module: nn.Module, n_expression: int) -> str:  # возвращает строковый путь к первому линейному слою
    if not hasattr(module, "z_encoder"):
        raise ValueError("The module has no z_encoder; this is not a supported scVI encoder.")
    candidates: list[tuple[str, nn.Linear]] = []
    for local_name, child in module.z_encoder.named_modules():
        if isinstance(child, nn.Linear) and child.in_features >= n_expression:
            full_name = "z_encoder" if not local_name else f"z_encoder.{local_name}"
            candidates.append((full_name, child))
    if not candidates:
        raise ValueError(
            "Could not find a z_encoder linear receiving the expression panel "
            f"({n_expression} inputs)."
        )
    min_depth = min(name.count(".") for name, _ in candidates)
    shallow = [(name, child) for name, child in candidates if name.count(".") == min_depth]
    if len(shallow) != 1:
        readable = ", ".join(name for name, _ in shallow)
        raise ValueError(
            "Ambiguous expression-facing z_encoder layer: " + readable + ". "
            "Pass AdapterConfig(target_path=...) explicitly."
        )
    return shallow[0][0]


@dataclass(frozen=True)
class AdapterMetadata:
    format: str                  # защита от загрузки чекпоинта не того типа
    kind: AdapterKind            # тип адаптера, ориентир только affine_w0
    target_path: str             # 
    expression_width: int
    original_in_features: int
    original_out_features: int
    has_bias: bool
    train_bias: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "format": self.format,
            "kind": self.kind,
            "target_path": self.target_path,
            "expression_width": self.expression_width,
            "original_in_features": self.original_in_features,
            "original_out_features": self.original_out_features,
            "has_bias": self.has_bias,
            "train_bias": self.train_bias,
        }


class ExpressionLayerAdapter(nn.Module):
    """A frozen linear layer plus one permitted phase-I input-side update.

    If the reference model has encoder covariates, only the first
    ``expression_width`` columns are adapted. Covariate columns remain frozen
    by construction, which is critical for a broad model where they may encode
    study/tissue metadata.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        expression_width: int,
        config: AdapterConfig,
        target_path: str,
    ) -> None:
        super().__init__()
        if expression_width < 1 or expression_width > base.in_features:
            raise ValueError(
                f"expression_width={expression_width} is incompatible with "
                f"Linear(in_features={base.in_features})."
            )
        config.validate()
        self.kind: AdapterKind = config.kind
        self.expression_width = int(expression_width)
        self.original_in_features = int(base.in_features)
        self.original_out_features = int(base.out_features)
        self.target_path = target_path
        self.train_bias = bool(config.train_bias and base.bias is not None)

        self.register_buffer("base_weight", base.weight.detach().clone())
        if base.bias is None:
            self.register_buffer("base_bias", None)
        else:
            self.register_buffer("base_bias", base.bias.detach().clone())

        output_width = base.out_features
        if self.kind == "full_w0":
            self.delta_weight = nn.Parameter(
                torch.zeros(output_width, expression_width, dtype=base.weight.dtype)
            )
        elif self.kind == "affine_w0":
            self.log_scale = nn.Parameter(torch.zeros(expression_width, dtype=base.weight.dtype))
            self.shift = nn.Parameter(torch.zeros(expression_width, dtype=base.weight.dtype))
        else:
            raise ValueError(f"Unsupported adapter kind: {self.kind}")

        if self.train_bias:
            self.delta_bias = nn.Parameter(torch.zeros(output_width, dtype=base.weight.dtype))

    @property
    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            format="scgrip_expression_layer_adapter_v1",
            kind=self.kind,
            target_path=self.target_path,
            expression_width=self.expression_width,
            original_in_features=self.original_in_features,
            original_out_features=self.original_out_features,
            has_bias=self.base_bias is not None,
            train_bias=self.train_bias,
        )

    def _weight_delta(self) -> torch.Tensor:
        if self.kind == "full_w0":
            return self.delta_weight
        return torch.zeros_like(self.base_weight[:, : self.expression_width])

    def effective_expression_map(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        base_expression_weight = self.base_weight[:, : self.expression_width]
        weight = base_expression_weight + self._weight_delta()
        bias = self.base_bias
        if self.kind == "affine_w0":
            scale = self.log_scale.exp()
            weight = weight * scale.unsqueeze(0)
            # ``shift`` is added before the original linear map, whereas
            # ``scale`` changes its incoming coordinate. The two effects must
            # not be multiplied together in the induced bias.
            affine_bias = base_expression_weight @ self.shift
            bias = affine_bias if bias is None else bias + affine_bias
        if self.train_bias:
            bias = self.delta_bias if bias is None else bias + self.delta_bias
        return weight, bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.original_in_features:
            raise RuntimeError(
                f"{self.target_path} received {inputs.shape[-1]} features, expected "
                f"{self.original_in_features}."
            )
        expression = inputs[..., : self.expression_width]
        if self.kind == "affine_w0":
            expression = expression * self.log_scale.exp() + self.shift
            weight = self.base_weight[:, : self.expression_width]
            bias = self.base_bias
            if self.train_bias:
                bias = self.delta_bias if bias is None else bias + self.delta_bias
        else:
            weight, bias = self.effective_expression_map()
        out = F.linear(expression, weight, bias)
        if self.expression_width < self.original_in_features:
            out = out + F.linear(
                inputs[..., self.expression_width :],
                self.base_weight[:, self.expression_width :],
                None,
            )
        return out

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.parameters()

    def is_identity(self, *, atol: float = 0.0) -> bool:
        """Check the algebraic no-op state without relying on a data batch."""

        candidates: list[torch.Tensor] = []
        if self.kind == "full_w0":
            candidates.append(self.delta_weight)
        else:
            candidates.extend((self.log_scale, self.shift))
        if self.train_bias:
            candidates.append(self.delta_bias)
        return all(bool(torch.all(tensor.detach().abs() <= atol)) for tensor in candidates)

    def l2_penalty(self) -> torch.Tensor:
        """A scale-aware optional regulariser on the actual permitted update."""

        if self.kind == "affine_w0":
            values = (self.log_scale, self.shift)
        else:
            values = (self.delta_weight,)
        total = sum(value.square().mean() for value in values)
        if self.train_bias:
            total = total + self.delta_bias.square().mean()
        return total

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Serialize only update parameters, never a duplicate frozen base W0."""

        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.named_parameters()
        }

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        expected = {name for name, _ in self.named_parameters()}
        received = set(state)
        if expected != received:
            raise ValueError(
                f"Adapter state keys do not match. expected={sorted(expected)} "
                f"received={sorted(received)}"
            )
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                value = state[name]
                if tuple(value.shape) != tuple(parameter.shape):
                    raise ValueError(
                        f"Shape mismatch for {name}: checkpoint {tuple(value.shape)}, "
                        f"module {tuple(parameter.shape)}."
                    )
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def attach_expression_adapter(
    module: nn.Module,
    *,
    n_expression: int,
    config: AdapterConfig,
) -> ExpressionLayerAdapter:
    config.validate()
    for parameter in module.parameters():
        parameter.requires_grad_(False)

    path = config.target_path or infer_expression_layer_path(module, n_expression)
    parent, name = _parent_and_name(module, path)
    original = _get_child(parent, name)
    if not isinstance(original, nn.Linear):
        raise ValueError(
            f"Adapter target {path!r} is {type(original).__name__}, not torch.nn.Linear."
        )
    adapter = ExpressionLayerAdapter(
        original,
        expression_width=n_expression,
        config=config,
        target_path=path,
    ).to(device=original.weight.device, dtype=original.weight.dtype)
    _set_child(parent, name, adapter)

    trainable = [name for name, parameter in module.named_parameters() if parameter.requires_grad]
    expected = {f"{path}.{name}" for name, _ in adapter.named_parameters()}
    if set(trainable) != expected:
        raise AssertionError(
            "Phase-I trainable parameter audit failed. "
            f"expected={sorted(expected)}, actual={sorted(trainable)}"
        )
    return adapter


def attach_from_metadata(module: nn.Module, metadata: dict[str, object]) -> ExpressionLayerAdapter:

    required = {
        "format", "kind", "target_path", "expression_width",
        "original_in_features", "original_out_features", "has_bias",
        "train_bias",
    }
    missing = required.difference(metadata)
    if missing:
        raise ValueError(f"Checkpoint adapter metadata misses: {sorted(missing)}")
    if metadata["format"] != "scgrip_expression_layer_adapter_v1":
        raise ValueError(f"Unsupported adapter checkpoint format: {metadata['format']!r}")
    config = AdapterConfig(
        kind=str(metadata["kind"]),  # type: ignore[arg-type]
        train_bias=bool(metadata["train_bias"]),
        target_path=str(metadata["target_path"]),
    )
    adapter = attach_expression_adapter(
        module,
        n_expression=int(metadata["expression_width"]),
        config=config,
    )
    if (
        adapter.original_in_features != int(metadata["original_in_features"])
        or adapter.original_out_features != int(metadata["original_out_features"])
        or (adapter.base_bias is not None) != bool(metadata["has_bias"])
    ):
        raise ValueError("Base model W0 shape/bias does not match the adapter checkpoint.")
    return adapter
