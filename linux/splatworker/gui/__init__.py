"""Desktop front end for SplatWork (GTK 4 + libadwaita), packaged as a Flatpak.

Everything in this package except :mod:`splatworker.gui.gtk` is plain Python with no
GTK dependency, so the setup planner, pip progress parser, NVML wrapper, settings
store and server lifecycle controller can be unit tested on any machine.

The GUI never imports torch. The worker itself (``python -m splatworker serve``) runs
as a child process using the GPU runtime the setup assistant installed into
``$XDG_DATA_HOME/splatworker`` (see :mod:`splatworker.gui.runtime_plan` for why).
"""

APP_ID = "com.example.SplatWorker"
APP_NAME = "SplatWork"
