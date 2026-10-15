---
name: Replit Streamlit preview
description: Forwarded Preview setup for the root-level Streamlit app in this workspace.
---

For the root-level Streamlit workflow, binding to `0.0.0.0:5000` alone does not make the proxied Preview work. This workspace's forwarded Preview uses local port 5000 mapped to external port 3000, and Streamlit must allow the proxy's cross-origin WebSocket with `--server.enableCORS false`. Keep XSRF protection enabled. Re-check the validated port mapping after workflow changes or restarts; those operations may remove the standalone app mapping while retaining artifact service mappings.

**Why:** The local Streamlit health endpoint worked while the forwarded Preview returned 502; workflow logs showed rejected proxy WebSocket origins. The app opened after adding the mapping and CORS flag.

**How to apply:** For future Preview repairs of this app, inspect `.replit` and the workflow first, preserve the existing artifact port mappings, then verify both the forwarded page and Streamlit health endpoint.
