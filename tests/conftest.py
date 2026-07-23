import warnings

try:
    from pyparsing import PyparsingDeprecationWarning
except ImportError:
    PyparsingDeprecationWarning = DeprecationWarning

warnings.filterwarnings("ignore", category=PyparsingDeprecationWarning)


def pytest_configure(config):
    config.addinivalue_line(
        "filterwarnings",
        "ignore::pyparsing.util.PyparsingDeprecationWarning",
    )
