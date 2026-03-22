from fastapi import FastAPI

from api.routes import router
from core.config import settings
from core.logging_config import setup_logging

setup_logging(settings.LOG_LEVEL)

app = FastAPI(
	title=settings.APP_NAME,
	version=settings.APP_VERSION,
)

app.include_router(router)
