"""OpenEnv-compatible server entry point."""
from incident_commander_env.server.app import app  # noqa: F401


def main() -> None:
    """Start the environment server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
