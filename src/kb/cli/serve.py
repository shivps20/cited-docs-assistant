"""`kb serve`: the chat API and web UI."""

from kb.core.config import get_settings


def serve_command(args) -> int:
    """`kb serve`: start the chat API (models load once at startup, ~20 s)."""
    import ipaddress
    import os

    import uvicorn

    from kb.api.app import create_app
    from kb.api.users import UsersError, load_users

    os.environ["TQDM_DISABLE"] = "1"
    s = get_settings()
    host, port = args.host or s.api_host, args.port or s.api_port
    try:
        load_users(s.users_path)                    # report a bad users.yaml before loading models
    except UsersError as e:
        print(e)
        return 1
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host == "localhost"
    if not loopback:
        print(f"WARN listening on {host}: users are not authenticated (X-KB-User header); "
              "anyone who can reach this port can pick any user in users.yaml")
    from kb.core.perf import disable_power_throttling

    throttling = "off (full CPU speed in the background)" if disable_power_throttling() else "not changed"
    print(f"Windows power throttling: {throttling}")
    print(f"loading models; then serving on http://{host}:{port}  (health: /api/health, docs: /docs)", flush=True)
    uvicorn.run(create_app(), host=host, port=port, log_level="info")
    return 0
