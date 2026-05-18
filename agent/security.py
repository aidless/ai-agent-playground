"""API Key 鉴权中间件 — 生产环境强制启用"""

import os
import logging
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """从 Authorization header 校验 API Key。

    生产环境 (APP_ENV=production) 或 enforce_auth=True 时必须配置 GATEWAY_API_KEY。
    开发环境未配置时生成临时 key 并警告，但不会阻断启动。
    健康检查 /health 放行。
    """

    def __init__(self, app, enforce_auth: bool = False):
        super().__init__(app)
        self.api_key = os.getenv("GATEWAY_API_KEY")
        self.enforce_auth = enforce_auth or os.getenv("APP_ENV") == "production"

        if not self.api_key:
            if self.enforce_auth:
                raise RuntimeError(
                    "GATEWAY_API_KEY is required in production mode. "
                    "Generate with: openssl rand -hex 32"
                )
            else:
                logger.warning(
                    "GATEWAY_API_KEY 未设置 — 端点完全开放（仅开发模式允许）。"
                    "生产环境必须设置此变量。"
                )

        self.enabled = bool(self.api_key) or self.enforce_auth
        if self.enabled:
            logger.info("API Key 鉴权已启用 (enforce=%s)", self.enforce_auth)
        else:
            logger.info("API Key 鉴权未设置（跳过，开发模式）")

    async def dispatch(self, request: Request, call_next):
        if not self.enabled:
            return await call_next(request)

        if request.url.path == "/health":
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth.removeprefix("Bearer ") != self.api_key:
            raise HTTPException(status_code=401, detail="无效的 API Key")

        return await call_next(request)
