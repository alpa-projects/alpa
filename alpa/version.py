"""Version information."""
from jax._src.lib import xla_extension as xe

__version__ = "1.0.0.dev0"

# Check the minimal requirement of alpa's jaxlib.
minimal_alpa_jaxlib_version = (0, 1, 1)

try:
    alpa_jaxlib_version_str = xe.get_alpa_jaxlib_version()
    alpa_jaxlib_version = tuple(
        int(x) for x in alpa_jaxlib_version_str.split("."))
except AttributeError:
    alpa_jaxlib_version = (0, 0, 0)

if alpa_jaxlib_version < minimal_alpa_jaxlib_version:
    minimal_alpa_jaxlib_version_str = ".".join(
        str(x) for x in minimal_alpa_jaxlib_version)
    alpa_jaxlib_version_str = ".".join(str(x) for x in alpa_jaxlib_version)
    raise RuntimeError(
        f"The alpa-jaxlib's internal version is v{alpa_jaxlib_version_str}, "
        f"but the minimal requirement is v{minimal_alpa_jaxlib_version_str}. "
        f"Please update your tensorflow-alpa submodule and re-compile jaxlib. "
        f"Help : https://alpa-projects.github.io/developer/developer_guide.html"
        f"#updating-submodule-tensorflow-alpa.")
