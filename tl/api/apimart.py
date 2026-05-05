import asyncio
import base64
import os
from typing import Any

import aiohttp

from ..api_types import APIError, ApiRequestConfig
from .base import ApiProvider, ProviderRequest

class ApimartProvider:
    name: str = "apimart"

    async def build_request(
        self, *, client: Any, config: ApiRequestConfig
    ) -> ProviderRequest:
        api_base = (config.api_base or "https://api.apimart.ai").rstrip("/")
        url = f"{api_base}/v1/images/generations"

        api_key = config.api_key
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": config.model or "gpt-image-2",
            "prompt": config.prompt,
            "n": 1,
        }

        if config.aspect_ratio:
            payload["size"] = config.aspect_ratio

        if config.resolution:
            payload["resolution"] = config.resolution.lower()

        if config.reference_images:
            # Apimart allows up to 16 reference images
            processed_urls = []
            for img in config.reference_images[:16]:
                img_str = str(img).strip()
                if img_str.startswith(("http://", "https://", "data:image/")):
                    processed_urls.append(img_str)
                elif os.path.isfile(img_str):
                    # It's a local file path, read and encode it
                    try:
                        with open(img_str, "rb") as f:
                            b64_data = base64.b64encode(f.read()).decode("utf-8")
                        # Guess mime type simply from extension, default to png
                        ext = os.path.splitext(img_str)[1].lower()
                        mime_type = f"image/{ext[1:]}" if ext in [".png", ".jpg", ".jpeg", ".webp"] else "image/png"
                        processed_urls.append(f"data:{mime_type};base64,{b64_data}")
                    except Exception:
                        pass # Ignore file read errors
                else:
                    # It's likely a raw base64 string, prepend the data URI scheme
                    # Assuming png as default if we don't know
                    processed_urls.append(f"data:image/png;base64,{img_str}")
            
            if processed_urls:
                payload["image_urls"] = processed_urls

        return ProviderRequest(url=url, headers=headers, payload=payload)

    async def parse_response(
        self,
        *,
        client: Any,
        response_data: dict[str, Any],
        session: aiohttp.ClientSession,
        api_base: str | None = None,
        http_status: int | None = None,
    ) -> tuple[list[str], list[str], str | None, str | None]:
        try:
            task_id = response_data["data"][0]["task_id"]
        except (KeyError, IndexError, TypeError) as e:
            raise APIError(
                f"apimart 响应格式异常，未找到 task_id: {response_data}",
                error_type="invalid_response"
            ) from e

        base = (api_base or "https://api.apimart.ai").rstrip("/")
        task_url = f"{base}/v1/tasks/{task_id}"

        # Fetch API Key for polling using the client method
        api_key = await client.get_next_api_key()
        headers = {"Authorization": f"Bearer {api_key}"}

        # 从配置中获取轮询参数，如果未配置或不是字典，则使用默认值
        provider_config = getattr(client, "_plugin_config", {}).get("api_settings", {}).get("provider_overrides", {})
        if isinstance(provider_config, list):
             # Try to find apimart config in the list of dicts
             apimart_cfg = next((c for c in provider_config if c.get("id") == "apimart"), {})
        else:
             apimart_cfg = provider_config.get("apimart", {}) if isinstance(provider_config, dict) else {}

        # AstrBot passes plugin config via client._plugin_config in some architectures, 
        # but to be safe, we can also extract it from kwargs or use defaults
        initial_wait = int(apimart_cfg.get("poll_initial_wait", 15))
        poll_interval = int(apimart_cfg.get("poll_interval", 5))
        max_attempts = int(apimart_cfg.get("poll_max_attempts", 60))

        # Wait before first polling
        await asyncio.sleep(initial_wait)

        for attempt in range(max_attempts):
            async with session.get(task_url, headers=headers) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise APIError(
                        f"apimart 轮询失败: HTTP {resp.status} - {error_text}",
                        resp.status,
                        retryable=False
                    )
                try:
                    data = await resp.json()
                except Exception as e:
                    raise APIError(
                        "apimart 轮询响应解析失败",
                        resp.status,
                        retryable=False
                    ) from e

            status = data.get("data", {}).get("status")

            if status == "completed":
                try:
                    image_url = data["data"]["result"]["images"][0]["url"][0]
                    
                    # Download image to local
                    from ..tl_utils import save_image_data
                    async with session.get(image_url) as img_resp:
                        if img_resp.status == 200:
                            img_data = await img_resp.read()
                            local_path = await save_image_data(img_data)
                            if local_path:
                                return [], [local_path], None, None
                    
                    return [image_url], [], None, None
                except (KeyError, IndexError) as e:
                    raise APIError(
                        f"apimart completed 但图片路径异常: {data}",
                        error_type="invalid_response"
                    ) from e

            if status == "failed":
                error_msg = data.get("data", {}).get("error", {}).get("message", "未知错误")
                raise APIError(f"apimart 生图失败: {error_msg}", retryable=False)

            # status is submitted or processing, continue polling
            await asyncio.sleep(poll_interval)

        raise APIError(
            f"apimart 生图超时，task_id={task_id}",
            error_type="timeout",
            retryable=True
        )
