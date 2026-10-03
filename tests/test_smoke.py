from credit_scoring import config


def test_config_lists_ten_raw_features():
    assert len(config.RAW_FEATURES) == 10
    assert set(config.LATE_COLUMNS) <= set(config.RAW_FEATURES)
