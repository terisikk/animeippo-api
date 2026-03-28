import structlog

from animeippo.logging import configure_logging


def test_configure_logging_returns_log_level(monkeypatch):
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    log_level = configure_logging()

    assert log_level == "DEBUG"


def test_configure_logging_production_mode(monkeypatch):
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.delenv("LOG_LEVEL", raising=False)

    log_level = configure_logging()

    assert log_level == "INFO"


def test_structlog_produces_output_after_configure(monkeypatch):
    monkeypatch.setenv("DEBUG", "true")
    configure_logging()

    logger = structlog.get_logger()
    # Should not raise
    logger.info("test_event", key="value")
