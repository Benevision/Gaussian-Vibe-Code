"""'How to connect from the Mac' page."""

from __future__ import annotations

from gi.repository import Adw, Gtk

from .common import action_row, copy_text, icon_button, label

FIREWALL_UFW = ("sudo ufw allow from 192.168.0.0/16 to any port {port} proto tcp\n"
                "sudo ufw allow from 192.168.0.0/16 to any port 5353 proto udp")
# No --zone: the rule goes into the *default* zone (public / FedoraWorkstation / …), which is the one
# the network interface actually uses on most systems ("home" is rarely active).
FIREWALL_FIREWALLD = ("sudo firewall-cmd --permanent --add-port={port}/tcp --add-service=mdns\n"
                      "sudo firewall-cmd --reload")


class HelpPage(Gtk.ScrolledWindow):
    def __init__(self, toasts: Adw.ToastOverlay):
        super().__init__(hscrollbar_policy=Gtk.PolicyType.NEVER, vexpand=True)
        self.toasts = toasts
        self.page = Adw.PreferencesPage()
        self.set_child(self.page)
        self._groups = []
        self.update(addresses=[], port=7878, name="this computer", token_required=True, mdns=True)

    def _code(self, text: str) -> Gtk.Widget:
        box = Gtk.Box(spacing=6)
        lab = label(text, ("monospace", "card"), selectable=True)
        lab.set_hexpand(True)
        lab.set_margin_top(6)
        lab.set_margin_bottom(6)
        box.append(lab)
        box.append(icon_button("edit-copy-symbolic", "Copy", lambda *_: (copy_text(self, text),
                                                                         self._toast("Copied"))))
        return box

    def _toast(self, text: str) -> None:
        self.toasts.add_toast(Adw.Toast(title=text, timeout=2))

    def update(self, addresses, port: int, name: str, token_required: bool, mdns: bool) -> None:
        for g in self._groups:
            self.page.remove(g)
        self._groups = []
        addr = addresses[0] if addresses else "this computer's IP address"

        g1 = Adw.PreferencesGroup(title="1. Start the server here",
                                  description="Flip the big switch on the Status page. The status dot turns "
                                              "green when the Mac can connect.")
        g2 = Adw.PreferencesGroup(title="2. Open SplatWork on the Mac",
                                  description="Both computers must be on the same network (same Wi-Fi or "
                                              "wired LAN).")
        g2.add(action_row("Open Settings → Remote Workers.", icon="preferences-system-symbolic"))
        if mdns:
            g2.add(action_row(f"\"{name}\" appears in the list automatically after a few seconds. Select it.",
                              icon="network-wireless-symbolic"))
        g2.add(action_row(f"If it does not appear, choose Add Manually and enter http://{addr}:{port}",
                          icon="list-add-symbolic", selectable=True))
        if token_required:
            g2.add(action_row("When asked for the access token, use Copy next to the token on the Status page "
                              "and paste it on the Mac (or type it). The token stays the same until you "
                              "regenerate it.", icon="dialog-password-symbolic"))
        g3 = Adw.PreferencesGroup(title="3. Train",
                                  description="In a project, choose Remote Worker (or Both) as the training "
                                              "device. Jobs and their progress show up on the Jobs page here.")

        g4 = Adw.PreferencesGroup(
            title="The Mac does not see this computer?",
            description="Discovery uses Bonjour (multicast DNS on UDP port 5353). It does not cross routers, "
                        "VLANs or VPNs, and guest Wi-Fi networks often block it. Adding the worker by address "
                        "always works as long as TCP port {} is reachable.".format(port))
        g4.add(action_row("Check that a firewall allows TCP port {} and UDP port 5353 from your local "
                          "network. On Ubuntu (ufw):".format(port), icon="security-medium-symbolic"))
        g4.add(self._code(FIREWALL_UFW.format(port=port)))
        g4.add(action_row("On Fedora / openSUSE (firewalld):"))
        g4.add(self._code(FIREWALL_FIREWALLD.format(port=port)))
        g4.add(action_row("Test from the Mac's Terminal:", icon="utilities-terminal-symbolic"))
        g4.add(self._code(f"dns-sd -B _splatworker._tcp\ncurl http://{addr}:{port}/v1/info"))
        g4.add(action_row("A 401 error from curl is good news: the server is reachable and only asks for the "
                          "token."))

        g5 = Adw.PreferencesGroup(
            title="Security",
            description="The connection is plain HTTP inside your network. Keep the token secret, never "
                        "forward the port to the internet, and use a VPN such as Tailscale or WireGuard to "
                        "train from elsewhere.")
        g6 = Adw.PreferencesGroup(
            title="Running without the window",
            description="Servers without a desktop can run the same worker from a terminal or systemd unit: "
                        "flatpak run com.example.SplatWorker --headless")
        for g in (g1, g2, g3, g4, g5, g6):
            self.page.add(g)
            self._groups.append(g)
