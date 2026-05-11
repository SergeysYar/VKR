from __future__ import annotations

import ast
import random
from dataclasses import dataclass, field
from math import e, pi
from typing import Iterable, Mapping

Number = int | float
Scalar = Number | str

_ALLOWED_DISTRIBUTIONS = {"uniform", "normal"}
_ALLOWED_FUNCTIONS = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
}
_ALLOWED_CONSTANTS = {
    "pi": pi,
    "e": e,
}


def _to_float(value: object, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float))


def _validate_scalar_or_none(value: object, label: str) -> None:
    if value is None:
        return
    if _is_number(value) or isinstance(value, str):
        return
    raise TypeError(f"{label} must be int, float, str expression, or None.")


def _parse_expression(expr: str) -> ast.Expression:
    try:
        parsed = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression '{expr}'.") from exc
    return parsed


def _extract_dependencies(expr: str) -> set[str]:
    parsed = _parse_expression(expr)
    dependencies: set[str] = set()
    for node in ast.walk(parsed):
        if not isinstance(node, ast.Name):
            continue
        if node.id in _ALLOWED_FUNCTIONS or node.id in _ALLOWED_CONSTANTS:
            continue
        dependencies.add(node.id)
    return dependencies


def _eval_expression(expr: str, context: Mapping[str, float]) -> float:
    parsed = _parse_expression(expr)
    result = _eval_ast(parsed.body, context)
    return float(result)


def _eval_ast(node: ast.AST, context: Mapping[str, float]) -> float:
    if isinstance(node, ast.Constant):
        if _is_number(node.value):
            return float(node.value)
        raise ValueError("Only numeric constants are allowed in expressions.")

    if isinstance(node, ast.Num):
        return float(node.n)

    if isinstance(node, ast.Name):
        if node.id in context:
            return float(context[node.id])
        if node.id in _ALLOWED_CONSTANTS:
            return float(_ALLOWED_CONSTANTS[node.id])
        raise ValueError(f"Unknown symbol '{node.id}' in expression.")

    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left, context)
        right = _eval_ast(node.right, context)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            return left**right
        raise ValueError("Unsupported binary operation in expression.")

    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast(node.operand, context)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError("Unsupported unary operation in expression.")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are allowed in expressions.")
        function_name = node.func.id
        if function_name not in _ALLOWED_FUNCTIONS:
            raise ValueError(f"Unsupported function '{function_name}' in expression.")
        function = _ALLOWED_FUNCTIONS[function_name]
        args = [_eval_ast(arg, context) for arg in node.args]
        kwargs: dict[str, float] = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                raise ValueError("Keyword unpacking is not allowed in expressions.")
            kwargs[keyword.arg] = _eval_ast(keyword.value, context)
        return float(function(*args, **kwargs))

    raise ValueError("Unsupported expression element.")


def _resolve_scalar(value: Scalar | None, context: Mapping[str, float], label: str) -> float | None:
    if value is None:
        return None
    if _is_number(value):
        return float(value)
    if isinstance(value, str):
        expression = value.strip()
        if not expression:
            raise ValueError(f"{label} expression cannot be empty.")
        return _eval_expression(expression, context)
    raise TypeError(f"{label} must be int, float, str expression, or None.")


@dataclass
class Parameter:
    name: str
    value: Scalar | None = None
    min: Scalar | None = None
    max: Scalar | None = None
    distribution: str = "uniform"

    def __post_init__(self) -> None:
        self.name = self.name.strip()
        if not self.name:
            raise ValueError("Parameter name cannot be empty.")

        self.distribution = self.distribution.strip().lower()
        if self.distribution not in _ALLOWED_DISTRIBUTIONS:
            raise ValueError(
                f"distribution must be one of {sorted(_ALLOWED_DISTRIBUTIONS)}."
            )

        _validate_scalar_or_none(self.value, f"{self.name}.value")
        _validate_scalar_or_none(self.min, f"{self.name}.min")
        _validate_scalar_or_none(self.max, f"{self.name}.max")

        if _is_number(self.min) and _is_number(self.max):
            min_value = _to_float(self.min, f"{self.name}.min")
            max_value = _to_float(self.max, f"{self.name}.max")
            if min_value > max_value:
                raise ValueError(f"{self.name}: min cannot be greater than max.")

    def dependencies(self) -> set[str]:
        deps: set[str] = set()
        for raw in (self.value, self.min, self.max):
            if isinstance(raw, str):
                deps.update(_extract_dependencies(raw))
        return deps

    def sample(self, rng: random.Random, context: Mapping[str, float]) -> float:
        resolved_value = _resolve_scalar(self.value, context, f"{self.name}.value")
        resolved_min = _resolve_scalar(self.min, context, f"{self.name}.min")
        resolved_max = _resolve_scalar(self.max, context, f"{self.name}.max")

        if (
            resolved_min is not None
            and resolved_max is not None
            and resolved_min > resolved_max
        ):
            raise ValueError(f"{self.name}: evaluated min cannot be greater than max.")

        if self.distribution == "uniform":
            return self._sample_uniform(rng, resolved_value, resolved_min, resolved_max)
        if self.distribution == "normal":
            return self._sample_normal(rng, resolved_value, resolved_min, resolved_max)
        raise ValueError(f"Unsupported distribution '{self.distribution}'.")

    def _sample_uniform(
        self,
        rng: random.Random,
        resolved_value: float | None,
        resolved_min: float | None,
        resolved_max: float | None,
    ) -> float:
        if resolved_min is None and resolved_max is None:
            if resolved_value is None:
                raise ValueError(
                    f"{self.name}: uniform distribution requires value or min/max."
                )
            return resolved_value

        low = resolved_min
        high = resolved_max
        if low is None:
            low = resolved_value if resolved_value is not None else high
        if high is None:
            high = resolved_value if resolved_value is not None else low

        if low is None or high is None:
            raise ValueError(
                f"{self.name}: unable to resolve uniform bounds from value/min/max."
            )
        if low > high:
            raise ValueError(f"{self.name}: uniform low bound cannot exceed high bound.")
        if low == high:
            return low
        return rng.uniform(low, high)

    def _sample_normal(
        self,
        rng: random.Random,
        resolved_value: float | None,
        resolved_min: float | None,
        resolved_max: float | None,
    ) -> float:
        if resolved_min is not None and resolved_max is not None and resolved_min == resolved_max:
            return resolved_min

        if resolved_value is not None:
            mean = resolved_value
        elif resolved_min is not None and resolved_max is not None:
            mean = (resolved_min + resolved_max) / 2.0
        elif resolved_min is not None:
            mean = resolved_min
        elif resolved_max is not None:
            mean = resolved_max
        else:
            raise ValueError(
                f"{self.name}: normal distribution requires value or min/max."
            )

        if resolved_min is not None and resolved_max is not None:
            std_dev = max((resolved_max - resolved_min) / 6.0, 1e-12)
        else:
            std_dev = max(abs(mean) * 0.1, 1e-12)

        sample_value = rng.gauss(mean, std_dev)
        if resolved_min is not None:
            sample_value = max(sample_value, resolved_min)
        if resolved_max is not None:
            sample_value = min(sample_value, resolved_max)
        return sample_value


@dataclass
class ParameterSet:
    parameters: dict[str, Parameter] = field(default_factory=dict)
    seed: int | None = 0

    def __init__(
        self,
        parameters: Iterable[Parameter] | Mapping[str, object] | None = None,
        seed: int | None = 0,
    ) -> None:
        self.parameters = {}
        self.seed = seed
        self._rng = random.Random(seed)

        if parameters is None:
            return

        if isinstance(parameters, Mapping):
            for name, raw in parameters.items():
                self.parameters[name] = self._coerce_parameter(name, raw)
            return

        for parameter in parameters:
            self.add(parameter)

    def add(self, parameter: Parameter) -> None:
        if parameter.name in self.parameters:
            raise ValueError(f"Parameter '{parameter.name}' already exists.")
        self.parameters[parameter.name] = parameter

    def sample(self, seed: int | None = None) -> dict[str, float]:
        rng = random.Random(seed) if seed is not None else self._rng
        order = self._evaluation_order()

        sampled: dict[str, float] = {}
        for name in order:
            sampled[name] = self.parameters[name].sample(rng, sampled)
        return sampled

    def override(self, values: Mapping[str, object]) -> ParameterSet:
        for name, raw in values.items():
            if name in self.parameters:
                self.parameters[name] = self._override_parameter(self.parameters[name], raw)
            else:
                self.parameters[name] = self._coerce_parameter(name, raw)
        return self

    def _override_parameter(self, parameter: Parameter, raw: object) -> Parameter:
        if isinstance(raw, Parameter):
            if raw.name != parameter.name:
                return Parameter(
                    name=parameter.name,
                    value=raw.value,
                    min=raw.min,
                    max=raw.max,
                    distribution=raw.distribution,
                )
            return raw

        if isinstance(raw, Mapping):
            value = raw["value"] if "value" in raw else parameter.value
            min_value = raw["min"] if "min" in raw else parameter.min
            max_value = raw["max"] if "max" in raw else parameter.max
            distribution = (
                str(raw["distribution"])
                if "distribution" in raw
                else parameter.distribution
            )
            return Parameter(
                name=parameter.name,
                value=value,
                min=min_value,
                max=max_value,
                distribution=distribution,
            )

        if _is_number(raw):
            constant_value = float(raw)
            return Parameter(
                name=parameter.name,
                value=constant_value,
                min=constant_value,
                max=constant_value,
                distribution="uniform",
            )

        if isinstance(raw, str):
            expression = raw.strip()
            return Parameter(
                name=parameter.name,
                value=expression,
                min=None,
                max=None,
                distribution="uniform",
            )

        raise TypeError(f"Unsupported override value for parameter '{parameter.name}'.")

    def _coerce_parameter(self, name: str, raw: object) -> Parameter:
        if isinstance(raw, Parameter):
            if raw.name == name:
                return raw
            return Parameter(
                name=name,
                value=raw.value,
                min=raw.min,
                max=raw.max,
                distribution=raw.distribution,
            )

        if isinstance(raw, Mapping):
            distribution = str(raw["distribution"]) if "distribution" in raw else "uniform"
            return Parameter(
                name=name,
                value=raw.get("value"),
                min=raw.get("min"),
                max=raw.get("max"),
                distribution=distribution,
            )

        if _is_number(raw):
            constant_value = float(raw)
            return Parameter(
                name=name,
                value=constant_value,
                min=constant_value,
                max=constant_value,
                distribution="uniform",
            )

        if isinstance(raw, str):
            return Parameter(
                name=name,
                value=raw,
                distribution="uniform",
            )

        raise TypeError(f"Cannot convert value for parameter '{name}' to Parameter.")

    def _evaluation_order(self) -> list[str]:
        dependencies: dict[str, set[str]] = {}
        for name, parameter in self.parameters.items():
            deps = parameter.dependencies()
            unknown = deps - self.parameters.keys()
            if unknown:
                missing = ", ".join(sorted(unknown))
                raise ValueError(
                    f"Parameter '{name}' references unknown dependencies: {missing}."
                )
            dependencies[name] = set(deps)

        order: list[str] = []
        temporary: set[str] = set()
        permanent: set[str] = set()

        def visit(name: str) -> None:
            if name in permanent:
                return
            if name in temporary:
                raise ValueError(f"Cyclic parameter dependency detected at '{name}'.")

            temporary.add(name)
            for dep in sorted(dependencies[name]):
                visit(dep)
            temporary.remove(name)

            permanent.add(name)
            order.append(name)

        for name in self.parameters:
            visit(name)

        return order
