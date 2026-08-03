import warnings

try:
    from pyparsing import PyparsingDeprecationWarning
    _HAS_PYPARSING_DEPRECATION_WARNING = True
except ImportError:
    PyparsingDeprecationWarning = DeprecationWarning
    _HAS_PYPARSING_DEPRECATION_WARNING = False

warnings.filterwarnings("ignore", category=PyparsingDeprecationWarning)


def pytest_configure(config):
    # Newer pyparsing (>=3.2) removed PyparsingDeprecationWarning; only register
    # the string-based filter when the category actually resolves, otherwise
    # pytest errors out while parsing the warning configuration.
    if _HAS_PYPARSING_DEPRECATION_WARNING:
        config.addinivalue_line(
            "filterwarnings",
            "ignore::pyparsing.util.PyparsingDeprecationWarning",
        )
