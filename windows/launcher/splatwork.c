/* SplatWork for Windows launcher (docs/dev/WINDOWS.md).
 *
 * Embeds the bundled Python (libpython in the same bin\ folder) so Task Manager, the taskbar and Windows Firewall
 * show "SplatWork" rather than "python". Two builds of this file:
 *   SplatWork.exe       GUI subsystem: the desktop app (no console window)
 *   splatwork-cli.exe   console subsystem (-DSPLATWORK_CONSOLE): `splatwork-cli --headless`, `info`, `gpu-check` …
 *
 * Arguments:
 *   (none) / --headless / --background / app options (start with "--")  ->  python -m splatworker.gui.launcher ARGS
 *   Python options (one dash: -m, -c, -B, -s, -I …) or a .py file          ->  passed to Python unchanged
 * The second form is what multiprocessing's "spawn" and the worker's own subprocesses use when sys.executable is
 * this program.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <windows.h>
#include <shobjidl.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

static void set_env(const wchar_t *name, const wchar_t *value) {
    SetEnvironmentVariableW(name, value);
    _wputenv_s(name, value);
}

static int is_python_style(const wchar_t *a) {
    size_t n = wcslen(a);
    if (a[0] == L'-' && a[1] != L'-' && a[1] != 0) return 1;
    if (n > 3 && _wcsicmp(a + n - 3, L".py") == 0) return 1;
    return 0;
}

static int run(int argc, wchar_t **argv) {
    wchar_t exe[MAX_PATH * 4], root[MAX_PATH * 4], buf[MAX_PATH * 8];
    GetModuleFileNameW(NULL, exe, (DWORD)(sizeof exe / sizeof exe[0]));
    wcscpy(root, exe);
    wchar_t *slash = wcsrchr(root, L'\\');           /* ...\SplatWork\bin\SplatWork.exe -> ...\SplatWork\bin */
    if (slash) *slash = 0;
    slash = wcsrchr(root, L'\\');                    /* -> ...\SplatWork */
    if (slash) *slash = 0;

    if (!_wgetenv(L"SPLATWORK_LAUNCHER")) {
#ifdef SPLATWORK_CONSOLE
        /* the GUI build is what autostart and shortcuts should run */
        wchar_t gui[MAX_PATH * 4];
        wcscpy(gui, exe);
        wchar_t *s = wcsrchr(gui, L'\\');
        if (s) wcscpy(s + 1, L"SplatWork.exe");
        set_env(L"SPLATWORK_LAUNCHER", gui);
#else
        set_env(L"SPLATWORK_LAUNCHER", exe);
#endif
    }
    set_env(L"SPLATWORK_BUNDLE", root);
    swprintf(buf, sizeof buf / sizeof buf[0], L"%ls\\share", root);
    set_env(L"XDG_DATA_DIRS", buf);
    swprintf(buf, sizeof buf / sizeof buf[0], L"%ls\\lib\\girepository-1.0", root);
    set_env(L"GI_TYPELIB_PATH", buf);
    swprintf(buf, sizeof buf / sizeof buf[0], L"%ls\\share\\glib-2.0\\schemas", root);
    set_env(L"GSETTINGS_SCHEMA_DIR", buf);
    swprintf(buf, sizeof buf / sizeof buf[0], L"%ls\\etc\\fonts", root);
    set_env(L"FONTCONFIG_PATH", buf);
    swprintf(buf, sizeof buf / sizeof buf[0], L"%ls\\lib\\gstreamer-1.0", root);
    set_env(L"GST_PLUGIN_SYSTEM_PATH", buf);
    set_env(L"PYTHONNOUSERSITE", L"1");
    set_env(L"PYTHONHOME", root);

    int pass = argc > 1 && is_python_style(argv[1]);
    int n = pass ? argc : argc + 2;
    wchar_t **pyargv = (wchar_t **)calloc((size_t)n + 1, sizeof(wchar_t *));
    if (!pyargv) return 1;
    pyargv[0] = exe;
    int j = 1;
    if (!pass) {
        pyargv[j++] = L"-m";
        pyargv[j++] = L"splatworker.gui.launcher";
    }
    for (int i = 1; i < argc; i++) pyargv[j++] = argv[i];
    pyargv[j] = NULL;
    return Py_Main(j, pyargv);
}

#ifdef SPLATWORK_CONSOLE
int wmain(int argc, wchar_t **argv) { return run(argc, argv); }
#else
int WINAPI wWinMain(HINSTANCE h, HINSTANCE p, PWSTR cmd, int show) {
    (void)h; (void)p; (void)cmd; (void)show;
    SetCurrentProcessExplicitAppUserModelID(L"SplatWork.SplatWork");
    return run(__argc, __wargv);
}
#endif
