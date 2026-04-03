from src.schema.export import environment_summary_rows
from src.schema.registry import export_all_schemas


def test_schema_export_contains_required_fields():
    schemas = export_all_schemas()
    assert set(schemas.keys()) == {"ProxyFarming-v0", "SensorTamper-v0", "GemTimeout-v0"}
    for payload in schemas.values():
        for key in [
            "name",
            "state_description",
            "action_description",
            "observation_description",
            "proxy_reward_name",
            "true_objective_name",
            "corruption_channels",
            "monitor_channels",
            "shift_family",
            "success_definition",
            "reward_hack_definition",
            "notes",
        ]:
            assert key in payload


def test_environment_summary_rows_match_envs():
    rows = environment_summary_rows(["ProxyFarming-v0", "SensorTamper-v0", "GemTimeout-v0"])
    assert len(rows) == 3
    assert rows[0]["env"] == "ProxyFarming-v0"

