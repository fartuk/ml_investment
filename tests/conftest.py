import pytest

def pytest_addoption(parser):
    parser.addoption("--config_path", action="store", default="config_example.json")

def pytest_configure(config):
    pytest.config_path = config.getoption("config_path")

