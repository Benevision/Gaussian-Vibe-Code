"""GTK 4 + libadwaita user interface (needs PyGObject, GTK >= 4.10, libadwaita >= 1.4).

Only imported by :func:`splatworker.gui.launcher.main` when a window is wanted.
"""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
