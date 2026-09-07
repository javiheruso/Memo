from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.api.v1.router import api_router
from app.core.config import settings
from app.security import LoginRequired
from app.web.chat import router as chat_ui_router
from app.web.router import auth_router
from app.web.router import router as admin_ui_router

app = FastAPI(title=settings.app_name, version=settings.app_version)

app.include_router(api_router, prefix=settings.api_v1_prefix)
app.include_router(admin_ui_router)
app.include_router(auth_router)
app.include_router(chat_ui_router)


@app.exception_handler(LoginRequired)
async def _login_required_handler(request, exc: LoginRequired) -> RedirectResponse:
    return RedirectResponse(url="/admin/login", status_code=303)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
    }
