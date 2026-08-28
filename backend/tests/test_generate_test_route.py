from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).parents[2] / "scripts" / "generate_test_route.py"


def test_mock_visual_output_is_explicitly_labelled_as_non_real_terrain():
    spec = spec_from_file_location("generate_test_route", SCRIPT)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    response = SimpleNamespace(
        route_geojson={
            "geometry": {"coordinates": [[4.0, 45.0], [5.0, 45.0]]},
            "properties": {"terrain_source": "offline_mock_terrain"},
        },
        route=SimpleNamespace(score=SimpleNamespace(total_cost=1.0, terrain_cost=0.0)),
    )

    output = module.build_geojson(response, real_terrain=False)

    assert output["properties"]["terrain_source"] == "offline_mock_terrain"
    assert output["properties"]["real_terrain"] is False
