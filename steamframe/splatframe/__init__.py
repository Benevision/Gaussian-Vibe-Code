"""Splat Capture for Steam Frame — experimental on-headset capture companion.

Records stereo (IR tracking cameras and/or the optional RGB add-on) with
OpenXR head poses and IR-stereo depth into v2 ``.splatcapture`` bundles
(see ``docs/PROTOCOL.md`` §1, §8) and sends them to Splat Studio (§5).

IMPORTANT: Valve has not published a camera-access API for the Steam Frame
(as of Oct 2026).  Everything hardware-specific in here (device nodes,
pixel formats, camera placement, OpenXR runtime behaviour) is *unverified*
and goes through pluggable, user-configurable backends.
"""

__version__ = "0.1.0"
APP_NAME = "Splat Capture for Steam Frame"
