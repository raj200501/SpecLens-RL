from src.utils.system import profile_from_preflight


def test_profile_selection_full_safe():
    preflight = {
        "system": {"free_disk_bytes": 10 * 1024**3, "logical_cpu_cores": 8},
        "packages": {
            "torch": {"available": True},
            "gymnasium": {"available": True},
            "minigrid": {"available": True},
        },
    }
    assert profile_from_preflight(preflight) == "PROFILE_FULL_SAFE"


def test_profile_selection_scaffold_only_when_packages_missing():
    preflight = {
        "system": {"free_disk_bytes": 10 * 1024**3, "logical_cpu_cores": 8},
        "packages": {
            "torch": {"available": False},
            "gymnasium": {"available": True},
            "minigrid": {"available": True},
        },
    }
    assert profile_from_preflight(preflight) == "PROFILE_SCAFFOLD_ONLY"

