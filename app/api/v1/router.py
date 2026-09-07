from fastapi import APIRouter

from app.api.v1.endpoints import admin, chat, health, voice

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(admin.router)
api_router.include_router(chat.router)
api_router.include_router(voice.router)
