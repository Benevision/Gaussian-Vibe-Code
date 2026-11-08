from splatworker.gui import updates as U


def test_busy_jobs_counts_queued_and_running():
    jobs = [{"state": "completed"}, {"state": "queued"}, {"state": "training"}, {"state": "failed"}]
    assert [j["state"] for j in U.busy_jobs(jobs)] == ["queued", "training"]
    assert U.busy_jobs(None) == []


def test_next_action():
    assert U.next_action(U.AVAILABLE, auto=True, busy=False) == "update"
    assert U.next_action(U.INSTALLED, auto=True, busy=False) == "restart"
    assert U.next_action(U.AVAILABLE, auto=True, busy=True) is None
    assert U.next_action(U.INSTALLED, auto=False, busy=False) is None
    assert U.next_action(U.DOWNLOADING, auto=True, busy=False) is None


def test_state_from_monitor():
    assert U.state_from_monitor({"running-commit": "a", "local-commit": "a", "remote-commit": "b"}) == U.AVAILABLE
    assert U.state_from_monitor({"running-commit": "a", "local-commit": "b", "remote-commit": "b"}) == U.INSTALLED
    assert U.state_from_monitor({"running-commit": "a", "local-commit": "a", "remote-commit": "a"}) is None


def test_parse_progress():
    assert U.parse_progress({"status": 0, "n_ops": 2, "op": 1, "progress": 50}) == (U.DOWNLOADING, 0.75, "")
    assert U.parse_progress({"status": 2})[0] == U.INSTALLED
    assert U.parse_progress({"status": 1})[0] == U.IDLE
    st, _, err = U.parse_progress({"status": 3, "error_message": "no space"})
    assert st == U.FAILED and err == "no space"


def test_restart_argv_and_banner():
    assert U.restart_argv(False) == ["splatworker-app", "--after-update"]
    assert U.restart_argv(True)[-1] == "--background"
    title, button = U.banner_text(U.INSTALLED, "1.4.0", auto=True, busy=True)
    assert "1.4.0" in title and button == "Restart Now"
    assert U.banner_text(U.IDLE, "", True, False) == ("", None)


def test_auto_update_setting_roundtrip():
    from splatworker.gui.settings import AppSettings
    assert AppSettings().auto_update is True
    assert AppSettings.from_dict({"auto_update": False}).auto_update is False
    assert AppSettings.from_dict({"auto_update": "yes"}).auto_update is True
