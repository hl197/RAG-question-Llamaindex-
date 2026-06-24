"""
RAG 智能问答助手 — FastAPI 后端

启动：python server.py
访问：http://localhost:7860

API 端点：
  GET  /                    → 前端 SPA 页面
  POST /api/chat            → JSON 对话
  POST /api/chat/stream     → SSE 流式对话
  GET  /api/sessions        → 会话列表
  POST /api/sessions        → 创建新会话
  DELETE /api/sessions/{id} → 删除会话
  GET  /api/sessions/{id}/history → 会话历史
  POST /api/files/upload    → 上传文件
  GET  /api/files           → 文件列表
  DELETE /api/files/{name}  → 删除文件
  GET  /api/knowledge/stats → 知识库统计
  POST /api/knowledge/clear → 清空知识库
  GET  /api/usage           → Token 用量
"""

import asyncio
import json
import os
import re
import tempfile
import threading
import traceback
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

import config
from agent.rag_agent import RAGAgent

# ── 全局 Agent（线程安全） ──────────────────────────
_agent_lock = threading.Lock()
_agent: Optional[RAGAgent] = None


def get_agent() -> RAGAgent:
    """获取全局 RAGAgent 单例（延迟初始化 + 双重检查锁）"""
    global _agent
    if _agent is None:
        with _agent_lock:
            if _agent is None:
                config.setup_logging()
                _agent = RAGAgent()
    return _agent


# ── FastAPI 应用 ──────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时预热 Agent，关闭时清理资源"""
    get_agent()  # 预热（阻塞直到初始化完成）
    # 初始化完成后自动打开浏览器
    import webbrowser
    webbrowser.open(f"http://localhost:{config.SERVER_PORT}")
    yield


app = FastAPI(
    title="RAG 智能问答助手",
    description="基于 LlamaIndex + DeepSeek + ChromaDB 的检索增强生成系统",
    version="2.0.0",
    lifespan=lifespan,
)

# 静态文件（前端 SPA）
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ── 首页 ──────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端 SPA 页面"""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse(
        "<h1>RAG 智能问答助手</h1><p>前端页面未找到，请确保 static/index.html 存在。</p>",
        status_code=404,
    )


# ── 对话 API ──────────────────────────────────────


@app.post("/api/chat")
async def chat(payload: dict):
    """
    同步对话（返回完整回复 JSON）。

    Body: {"message": "...", "session_id": "..."}
    """
    message = (payload.get("message") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="消息不能为空")

    session_id = payload.get("session_id") or None
    agent = get_agent()

    loop = asyncio.get_event_loop()
    reply = await loop.run_in_executor(None, lambda: agent.chat(message, session_id))

    return {
        "reply": reply,
        "session_id": agent.get_current_session_id(),
    }


@app.post("/api/chat/stream")
async def chat_stream(payload: dict):
    """
    SSE 流式对话。

    Body: {"message": "...", "session_id": "..."}

    返回 text/event-stream，每个事件携带增量文本 token。
    """

    message = (payload.get("message") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="消息不能为空")

    session_id = payload.get("session_id") or None
    agent = get_agent()

    async def event_stream():
        try:
            loop = asyncio.get_event_loop()
            # 在线程池中运行同步生成器（返回 (event_type, data) 元组）
            gen = await loop.run_in_executor(
                None, lambda: agent.chat_stream(message, session_id)
            )

            final_text = ""
            for event_type, data in gen:
                if event_type == "lifecycle":
                    # Agent 生命周期步骤
                    yield f"data: {json.dumps({'type': 'lifecycle', 'step': data}, ensure_ascii=False)}\n\n"
                elif event_type == "token":
                    # 增量 token（累积文本）
                    final_text = data
                    yield f"data: {json.dumps({'type': 'token', 'token': data}, ensure_ascii=False)}\n\n"
                elif event_type == "done":
                    # 结束信号
                    final_data = {
                        "type": "done",
                        "session_id": agent.get_current_session_id(),
                        "full_text": final_text,
                    }
                    yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)  # 让出控制权

        except Exception as e:
            error_data = {"type": "error", "error": str(e), "done": True}
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
        },
    )


# ── 会话管理 API ──────────────────────────────────


@app.get("/api/sessions")
async def list_sessions():
    """获取所有会话列表"""
    agent = get_agent()
    sessions = agent.get_sessions()
    return {"sessions": sessions}


@app.post("/api/sessions")
async def create_session():
    """创建新会话"""
    agent = get_agent()
    session_id = agent.new_session()
    return {"session_id": session_id}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除指定会话"""
    agent = get_agent()
    agent.delete_session(session_id)
    return {"ok": True}


@app.get("/api/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """获取指定会话的历史消息"""
    agent = get_agent()
    if not agent.memory.session_exists(session_id):
        raise HTTPException(status_code=404, detail="会话不存在")

    history = agent.get_session_history(session_id)
    return {"session_id": session_id, "history": history}


# ── 文件管理 API ──────────────────────────────────


@app.post("/api/files/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    上传文件到知识库。

    接收 multipart/form-data，字段名 "file"。
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="未选择文件")

    # 检查扩展名
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {ext}，允许: {', '.join(config.ALLOWED_EXTENSIONS)}",
        )

    # 检查文件大小
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > config.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"文件过大 ({size_mb:.1f}MB)，限制 {config.MAX_FILE_SIZE_MB}MB",
        )

    # 写入临时文件
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        agent = get_agent()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, agent.upload_file, tmp_path)

        if result.get("success"):
            return result
        else:
            error_msg = result.get("error", "上传失败")
            print(f"❌ 上传失败: {error_msg}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=error_msg)
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 上传异常: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理临时文件
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.get("/api/files")
async def list_files():
    """获取已上传文件列表"""
    agent = get_agent()
    files = agent.get_uploaded_files()
    return {"files": files}


@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    """从知识库删除指定文件"""
    agent = get_agent()
    ok = agent.delete_file(filename)
    if not ok:
        raise HTTPException(status_code=404, detail="文件不存在或删除失败")
    return {"ok": True}


# ── 知识库管理 API ────────────────────────────────


@app.get("/api/knowledge/stats")
async def knowledge_stats():
    """获取知识库统计信息"""
    agent = get_agent()
    stats = agent.get_knowledge_stats()
    return stats


@app.post("/api/knowledge/clear")
async def clear_knowledge():
    """清空知识库"""
    agent = get_agent()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, agent.clear_knowledge)
    return {"ok": True}


# ── 用量统计 API ──────────────────────────────────


@app.get("/api/usage")
async def token_usage(session_id: Optional[str] = None):
    """
    获取 Token 用量统计。

    Query: ?session_id=xxx （可选，不传则返回全局总量）
    """
    agent = get_agent()
    if session_id:
        usage = agent.get_session_token_usage(session_id)
    else:
        usage = agent.get_total_token_usage()
    return {"usage": usage}


# ── 启动入口 ──────────────────────────────────────


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=config.SERVER_PORT,
        reload=False,
        log_level="info",
    )
