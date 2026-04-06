"""Re-export the FastAPI app from the main package for OpenEnv validator compatibility."""
from incident_commander_env.server.app import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
