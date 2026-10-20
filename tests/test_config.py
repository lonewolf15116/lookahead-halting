from lookahead.config import REPO_ROOT, load_config


def test_model_config_has_required_keys():
    cfg = load_config("model")
    for key in ("repo_id", "local_dir", "dtype", "total_ut_steps"):
        assert key in cfg
    assert cfg["total_ut_steps"] == 4


def test_protocol_is_in_the_repo():
    assert (REPO_ROOT / "docs" / "PROTOCOL.md").exists()
