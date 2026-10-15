---
name: Python package installation
description: Python module and package-install behavior in this Replit workspace.
---

When Python package installation fails because the base Python module has no `pip` or the interpreter is externally managed, install a full Python toolchain module before retrying package installation. Keep the app's `requirements.txt` as the authoritative dependency list and check for installer-added duplicate entries.

**Why:** The workspace's base Python module lacked `pip` and rejected direct package installation; the full Python toolchain enabled the supported package installer.

**How to apply:** For future Python apps in this workspace, check the active Python module before changing dependency files or retrying package installation.
