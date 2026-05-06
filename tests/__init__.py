"""Test package marker.

Snapshots the genuine ``builtins.print`` at package-load time as a
defensive measure for the async_logger test fixture. The orchestrator
modules no longer install the async writer at import (it moved into
``main()``), but if a future change re-introduces import-time install
or any test ends up triggering it indirectly, the captured reference
lets the fixture restore the genuine builtin without recovering it
from a polluted state.
"""

import builtins

REAL_BUILTIN_PRINT = builtins.print
