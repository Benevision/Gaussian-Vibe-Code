"""Bonjour / mDNS advertisement of ``_splatworker._tcp`` (PROTOCOL.md §4)."""

from __future__ import annotations

import logging
import socket
from typing import List, Optional

log = logging.getLogger("splatworker.mdns")

SERVICE_TYPE = "_splatworker._tcp.local."


def local_ipv4_addresses() -> List[str]:
    addrs: List[str] = []
    try:
        import ifaddr  # dependency of zeroconf

        for adapter in ifaddr.get_adapters():
            for ip in adapter.ips:
                if isinstance(ip.ip, str) and not ip.ip.startswith(("127.", "169.254.")):
                    addrs.append(ip.ip)
    except Exception:
        pass
    if not addrs:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("10.255.255.255", 1))
                addrs.append(s.getsockname()[0])
        except OSError:
            pass
    # docker bridge addresses are rarely useful to the Mac; keep them last
    return sorted(set(addrs), key=lambda a: (a.startswith("172.17."), a))


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
        addrs = self.addresses or local_ipv4_addresses()
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
