from __future__ import annotations

from synthetic_factory.generators.biomes import BIOME_REGISTRY, get_biome_definition


def test_biome_registry_contains_core_modules() -> None:
    expected = {
        "workshop",
        "office",
        "refinery",
        "boiler",
        "storage",
        "electrical",
        "maintenance",
        "laboratory",
        "control",
    }

    assert expected.issubset(BIOME_REGISTRY.keys())
    for name in expected:
        definition = BIOME_REGISTRY[name]
        assert definition.name == name
        assert definition.primitives
        assert definition.rules


def test_unknown_biome_falls_back_to_workshop() -> None:
    fallback = get_biome_definition("unknown_new_biome")
    assert fallback.name == "workshop"
