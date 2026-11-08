"""Bonjour / mDNS advertisement of ``_splatworker._tcp`` (PROTOCOL.md §4)."""

from __future__ import annotations

import logging
import socket
from typing import List, Optional

log = logging.getLogger("splatworker.mdns")

SERVICE_TYPE = "_splatworker._tcp.local."


def interface_kind(name: str, sys_net: str = "/sys/class/net") -> str:
    """"ethernet", "wifi" or "virtual" for a network interface (Linux sysfs, else by name)."""
    import os

    base = os.path.join(sys_net, name)
    if os.path.isdir(base):
        if os.path.isdir(os.path.join(base, "wireless")) or os.path.isdir(os.path.join(base, "phy80211")):
            return "wifi"
        if not os.path.exists(os.path.join(base, "device")):  # bridges, docker, VPN tunnels, loopback
            return "virtual"
        return "ethernet"
    n = name.lower()
    if n.startswith(("wl", "wifi", "wlan")):
        return "wifi"
    if n.startswith(("en", "eth")):
        return "ethernet"
    return "virtual"


_KIND_ORDER = {"ethernet": 0, "wifi": 1, "virtual": 2}


def local_ipv4_interfaces() -> List[tuple]:
    """(address, kind) for each LAN IPv4 address, ethernet first, then Wi-Fi, then virtual adapters."""
    found: dict = {}
    try:
        import ifaddr  # dependency of zeroconf

        for adapter in ifaddr.get_adapters():
            kind = interface_kind(adapter.name)
            for ip in adapter.ips:
                if isinstance(ip.ip, str) and not ip.ip.startswith(("127.", "169.254.")):
                    found.setdefault(ip.ip, kind)
    except Exception:
        pass
    if not found:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("10.255.255.255", 1))
                found[s.getsockname()[0]] = "ethernet"
        except OSError:
            pass
    return sorted(found.items(), key=lambda a: (_KIND_ORDER.get(a[1], 3), a[0].startswith("172.17."), a[0]))


def local_ipv4_addresses() -> List[str]:
    return [a for a, _ in local_ipv4_interfaces()]


def addrs_txt(interfaces: List[tuple]) -> str:
    """TXT ``addrs``: preferred first, e.g. ``192.168.0.113:ethernet,192.168.0.120:wifi`` (worker >= 1.3.10)."""
    return ",".join(f"{a}:{k}" for a, k in interfaces if k != "virtual")[:240]


def txt_record(gpu_name: str, auth: bool) -> dict:
    return {"version": "1", "gpu": gpu_name[:200], "auth": "1" if auth else "0"}


class Advertiser:
    def __init__(self, name: str, port: int, gpu_name: str, auth: bool,
                 addresses: Optional[List[str]] = None):
        self.name = name
        self.port = port
        self.properties = txt_record(gpu_name, auth)
        self.addresses = addresses
        self._zc = None
        self._info = None

    def start(self) -> None:
        try:
            from zeroconf import IPVersion, ServiceInfo, Zeroconf
        except ImportError:
            log.warning("zeroconf not installed; mDNS advertisement disabled")
            return
        if self.addresses:
            addrs = self.addresses
        else:
            ifs = local_ipv4_interfaces()
            addrs = [a for a, _ in ifs]
            if addrs_txt(ifs):
                self.properties = {**self.properties, "addrs": addrs_txt(ifs)}
        if not addrs:
            log.warning("no LAN IPv4 address found; mDNS advertisement disabled")
            return
        # instance names may not contain dots
        instance = self.name.replace(".", "-")[:63]
        self._info = ServiceInfo(
            SERVICE_TYPE,
            f"{instance}.{SERVICE_TYPE}",
            port=self.port,
            properties=self.properties,
            server=f"{socket.gethostname().split('.')[0]}.local.",
            parsed_addresses=addrs,
        )
        self._zc = Zeroconf(ip_version=IPVersion.V4Only)
        self._zc.register_service(self._info, allow_name_change=True)
        log.info("advertising %s on %s port %d (%s)", instance, addrs, self.port, self.properties)

    def stop(self) -> None:
        if self._zc is not None:
            try:
                if self._info is not None:
                    self._zc.unregister_service(self._info)
            finally:
                self._zc.close()
                self._zc = None
