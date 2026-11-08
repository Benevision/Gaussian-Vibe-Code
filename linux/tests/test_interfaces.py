"""Ethernet is preferred over Wi-Fi when a worker is on both (Mac connects to the first address that answers)."""
from splatworker import discovery


def test_interface_kind_from_sysfs(tmp_path):
    (tmp_path / "enp5s0" / "device").mkdir(parents=True)
    (tmp_path / "wlp4s0" / "wireless").mkdir(parents=True)
    (tmp_path / "wlp4s0" / "device").mkdir()
    (tmp_path / "docker0").mkdir()
    assert discovery.interface_kind("enp5s0", str(tmp_path)) == "ethernet"
    assert discovery.interface_kind("wlp4s0", str(tmp_path)) == "wifi"
    assert discovery.interface_kind("docker0", str(tmp_path)) == "virtual"


def test_interface_kind_by_name_without_sysfs(tmp_path):
    assert discovery.interface_kind("eth0", str(tmp_path)) == "ethernet"
    assert discovery.interface_kind("wlan0", str(tmp_path)) == "wifi"


def test_ethernet_first(monkeypatch):
    class IP:
        def __init__(self, ip): self.ip = ip

    class A:
        def __init__(self, name, ip): self.name, self.ips = name, [IP(ip)]

    import types, sys
    fake = types.SimpleNamespace(get_adapters=lambda: [A("wlp4s0", "192.168.0.120"), A("docker0", "172.17.0.1"),
                                                       A("enp5s0", "192.168.0.113"), A("lo", "127.0.0.1")])
    monkeypatch.setitem(sys.modules, "ifaddr", fake)
    kinds = {"wlp4s0": "wifi", "docker0": "virtual", "enp5s0": "ethernet", "lo": "virtual"}
    monkeypatch.setattr(discovery, "interface_kind", lambda n: kinds[n])
    ifs = discovery.local_ipv4_interfaces()
    assert ifs[0] == ("192.168.0.113", "ethernet") and ifs[1] == ("192.168.0.120", "wifi")
    assert discovery.addrs_txt(ifs) == "192.168.0.113:ethernet,192.168.0.120:wifi"
