import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse
import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse, Response

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()
app = FastAPI()
APP_SECRET = os.getenv("APP_SECRET", "666")
APIKEY = os.getenv("OMN_APIKEY", "")
AUTHORIZATION = os.getenv("OMN_AUTHORIZATION", "")
USER_ID = os.getenv("OMN_USER_ID", "")

headers = {
    'accept': '*/*',
    'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
    'cache-control': 'no-cache',
    'origin': 'https://app.omnigpt.co',
    'pragma': 'no-cache',
    'priority': 'u=1, i',
    'referer': 'https://app.omnigpt.co/',
    'sec-ch-ua': '"Microsoft Edge";v="129", "Not=A?Brand";v="8", "Chromium";v="129"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-site',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36 Edg/129.0.0.0',
    'x-client-info': 'supabase-ssr/0.3.0',
}


ALLOWED_MODELS = [
    {"id": "gpt-4o-mini", "name": "gpt-4o-mini"},
    {"id": "gpt-4o", "name": "gpt-4o"},
    {"id": "gpt-4", "name": "gpt-4"},
    {"id": "gpt-4-turbo", "name": "gpt-4-turbo"},
    {"id": "o1-preview", "name": "o1-preview"},
    {"id": "o1-mini", "name": "o1-mini"},
    {"id": "claude_3.5_sonnet", "name": "claude_3.5_sonnet"},
    {"id": "claude_3_opus", "name": "claude_3_opus"},
    {"id": "deepseek-coder", "name": "deepseek-coder"},
    {"id": "llama-3-lumimaid-70b", "name": "llama-3-lumimaid-70b"},
    {"id": "toppy-m-7b:nitro", "name": "toppy-m-7b:nitro"},
    {"id": "gemini-pro-1.5", "name": "gemini-pro-1.5"},
]
# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源，您可以根据需要限制特定源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)
security = HTTPBearer()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False


def simulate_data(content, model):
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(datetime.now().timestamp()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content, "role": "assistant"},
                "finish_reason": None,
            }
        ],
        "usage": None,
    }


def stop_data(content, model):
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(datetime.now().timestamp()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content, "role": "assistant"},
                "finish_reason": "stop",
            }
        ],
        "usage": None,
    }
    
    
def create_chat_completion_data(content: str, model: str, finish_reason: Optional[str] = None) -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(datetime.now().timestamp()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content, "role": "assistant"},
                "finish_reason": finish_reason,
            }
        ],
        "usage": None,
    }


def verify_app_secret(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != APP_SECRET:
        raise HTTPException(status_code=403, detail="Invalid APP_SECRET")
    return credentials.credentials


@app.options("/hf/v1/chat/completions")
async def chat_completions_options():
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        },
    )


def replace_escaped_newlines(input_string: str) -> str:
    return input_string.replace("\\n", "\n")


async def create_conversation(apikey, authorization, thread_model, user_identifier):
    header_new = headers
    header_new['accept'] = 'application/vnd.pgrst.object+json'
    header_new['content-profile'] = 'public'
    header_new['prefer'] = 'return=representation'
    header_new['apikey'] = apikey
    header_new['authorization'] = f'Bearer {authorization}'
    data = {
        'thread_name': 'New Thread',
        'thread_model': thread_model,
        'user_identifier': user_identifier,
    }
    url = 'https://api.omnigpt.co/rest/v1/threads?select=*'
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=header_new, json=data, timeout=30)
            response.raise_for_status()  # Raises stored HTTPError, if one occurred.
            json_response = response.json()
            if json_response.get('thread_id'):
                logger.info("Deployment Conversation Created Successfully")
                return json_response.get('thread_id')
        except httpx.HTTPStatusError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
        except Exception as err:
            logger.error(f"An error occurred: {err}")


async def delete_conversation(thread_id, apikey, authorization):
    url = 'https://api.omnigpt.co/rest/v1/threads'
    header_new = headers
    header_new['apikey'] = apikey
    header_new['authorization'] = f'Bearer {authorization}'
    params = {
        'thread_id': f'eq.{thread_id}',
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.delete(url, headers=headers, params=params)
            response.raise_for_status()  # Raises stored HTTPError, if one occurred.
            logger.info("Delete Conversation Successfully")
        except httpx.HTTPStatusError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
        except Exception as err:
            logger.error(f"An error occurred: {err}")


@app.get("/hf/v1/models")
async def list_models():
    return {"object": "list", "data": ALLOWED_MODELS}


@app.post("/hf/v1/chat/completions")
async def chat_completions(
    request: ChatRequest, app_secret: str = Depends(verify_app_secret),
    raw_request: Request = None
):
    logger.info(f"Received chat completion request for model: {request.model}")

    if request.model not in [model['id'] for model in ALLOWED_MODELS]:
        raise HTTPException(
            status_code=400,
            detail=f"Model {request.model} is not allowed. Allowed models are: {', '.join(model['id'] for model in ALLOWED_MODELS)}",
        )


    thread_id = await create_conversation(APIKEY, AUTHORIZATION, request.model, USER_ID)
    # 使用 OpenAI API
    default_system_prompt = 'You are a helpful assistant. \nAdditional Context:\n- Respond using KaTeX to render mathematical expressions in your responses. Only use KaTeX for actual mathematical equations and formulas. For all other text, use regular formatting. Use the following format: $<expression>$ for inline math and $$<expression>$$ for display math.\n\nExample:\nUser: How do you find the roots of a quadratic equation?\nAssistant: To find the roots of a quadratic equation given by $ax^2 + bx + c = 0$, you can use the quadratic formula:\n$$x = \frac{-b pm sqrt{b^2 - 4ac}}{2a}$$\n'
    json_data = {
        'gptModel': request.model,
        'thread_id': thread_id,
        'system': default_system_prompt,
        "prompt": "\n".join(
            [
                f"{'User' if msg.role == 'user' else 'Assistant'}: {msg.content}"
                for msg in request.messages
            ]
        ),
        'files': [],
    }
    headers_new = headers
    headers_new['authorization'] = f'Bearer {AUTHORIZATION}'
    async def generate():
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream('POST', 'https://api.omnigpt.co/functions/v1/connect-llm', headers=headers_new, json=json_data, timeout=120.0) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line and ('[DONE]' not in line):
                            line_json = json.loads(line[6:])
                            delta = line_json["choices"][0].get("delta")
                            if delta:
                                yield f"data: {json.dumps(create_chat_completion_data(delta.get('content', ''), request.model))}\n\n"
                    yield f"data: {json.dumps(create_chat_completion_data('', request.model, 'stop'))}\n\n"
                    yield "data: [DONE]\n\n"
                    await delete_conversation(thread_id, APIKEY, AUTHORIZATION)
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error occurred: {e}")
                await delete_conversation(thread_id, APIKEY, AUTHORIZATION)
                raise HTTPException(status_code=e.response.status_code, detail=str(e))
            except httpx.RequestError as e:
                logger.error(f"An error occurred while requesting: {e}")
                await delete_conversation(thread_id, APIKEY, AUTHORIZATION)
                raise HTTPException(status_code=500, detail=str(e))


    if request.stream:
        logger.info("Streaming response")
        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        logger.info("Non-streaming response")
        full_response = ""
        async for chunk in generate():
            if chunk.startswith("data: ") and not chunk[6:].startswith("[DONE]"):
                # print(chunk)
                data = json.loads(chunk[6:])
                if data["choices"][0]["delta"].get("content"):
                    full_response += data["choices"][0]["delta"]["content"]
        
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": full_response},
                    "finish_reason": "stop",
                }
            ],
            "usage": None,
        }



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
