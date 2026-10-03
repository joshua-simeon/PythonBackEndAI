from fastapi import FastAPI

from routers.documents import router as documents_router
from routers.health import router as health_router

app = FastAPI(title="Document API", version="0.1.0")
app.include_router(documents_router)
app.include_router(health_router)
