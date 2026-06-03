import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_job_board_sites_accepts_boards_key():
    from applypilot import config

    assert config.job_board_sites({"boards": ["indeed", "linkedin"]}) == ["indeed", "linkedin"]


def test_job_board_sites_prefers_sites_key_for_compatibility():
    from applypilot import config

    search_cfg = {
        "sites": ["zip_recruiter"],
        "boards": ["indeed", "linkedin"],
    }

    assert config.job_board_sites(search_cfg) == ["zip_recruiter"]


def test_job_board_sites_returns_none_when_unconfigured():
    from applypilot import config

    assert config.job_board_sites({}) is None


def test_location_filters_accept_nested_location_config():
    from applypilot import config

    search_cfg = {
        "location": {
            "accept_patterns": ["San Francisco", "Remote"],
            "reject_patterns": ["onsite only"],
        }
    }

    assert config.location_filters(search_cfg) == (["San Francisco", "Remote"], ["onsite only"])


def test_location_filters_prefers_flat_legacy_keys():
    from applypilot import config

    search_cfg = {
        "location_accept": ["Bay Area"],
        "location_reject_non_remote": ["New York only"],
        "location": {
            "accept_patterns": ["Remote"],
            "reject_patterns": ["onsite only"],
        },
    }

    assert config.location_filters(search_cfg) == (["Bay Area"], ["New York only"])
