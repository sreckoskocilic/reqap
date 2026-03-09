from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.config import Settings
from app.llm.router import LLMRouter


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = Settings()
    app.state.config = cfg
    app.state.router = LLMRouter(cfg)
    yield


app = FastAPI(title="reqap — EPUB Review Service", lifespan=lifespan)
app.include_router(api_router)

# Serve the portal at /  (must be mounted last)
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
