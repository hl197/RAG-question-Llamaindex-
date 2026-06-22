"""
RAG 智能问答助手 — Gradio Web 界面

启动：python app.py
访问：http://localhost:7860
"""

import os

import tempfile
import threading
from typing import List, Optional

import gradio as gr

import config
from agent.rag_agent import RAGAgent

# ── 全局 Agent（线程安全） ──────────────────────
_agent_lock = threading.Lock()
_agent: Optional[RAGAgent] = None


def get_agent() -> RAGAgent:
    global _agent
    if _agent is None:
        with _agent_lock:
            if _agent is None:
                _agent = RAGAgent()
    return _agent


# ── UI 回调函数 ──────────────────────────────────

def chat_fn(message: str, history: List) -> str:
    """
    处理用户聊天消息。

    Args:
        message: 用户输入
        history: Gradio 聊天历史（自动维护）

    Returns:
        str: Agent 回复
    """
    agent = get_agent()

    # 确保有当前会话
    session_id = agent.get_current_session_id()
    if session_id is None:
        session_id = agent.new_session()

    # 发送消息
    reply = agent.chat(message, session_id=session_id)

    # 更新会话下拉框
    return reply


def load_session_history(session_label) -> List:
    """解析 session_label 并加载对应会话的历史消息"""
    if not session_label or session_label == "（无会话）":
        return []
    import re
    match = re.search(r'\(([a-f0-9]+)\)', session_label)
    if not match:
        return []
    session_id = match.group(1)
    agent = get_agent()
    return agent.get_session_history(session_id)


def initial_load():
    """页面加载时刷新数据并加载首个会话的历史到聊天框"""
    file_list_val = refresh_file_list()
    stats_val = refresh_stats()
    session_update = refresh_sessions()  # gr.update
    # 从 gr.update 提取当前选中的会话 label
    current_label = getattr(session_update, 'value', None)
    # 同步切换 agent 到对应的会话，并加载历史
    if current_label and current_label != "（无会话）":
        _, history = switch_session(current_label)
    else:
        history = []
    return file_list_val, stats_val, session_update, history


def upload_file(file_obj) -> str:
    """
    上传文件处理。

    Args:
        file_obj: Gradio 上传的文件对象（FileData，含 .path 和 .name）

    Returns:
        str: 提示消息
    """
    if file_obj is None:
        return "请选择要上传的文件"

    # Gradio 6 FileData: .path = 临时文件路径, .name = 原始文件名
    file_path = getattr(file_obj, "path", None) or getattr(file_obj, "name", None)
    if file_path is None:
        return "无法获取文件路径"

    # 检查文件大小
    try:
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > 50:
            return f"文件过大（{file_size_mb:.1f}MB），请上传小于 50MB 的文件"
    except OSError:
        pass  # 临时文件可能已清理，交给后续处理

    # 上传并索引
    agent = get_agent()
    result = agent.upload_file(file_path)

    if result["success"]:
        return f"✅ 已上传「{result['filename']}」，新增 {result.get('nodes_added', 0)} 个文档片段"
    else:
        return f"❌ 上传失败: {result.get('error', '未知错误')}"


def refresh_file_list() -> str:
    """刷新已上传文件列表显示"""
    agent = get_agent()
    files = agent.get_uploaded_files()
    if not files:
        return "（暂无文件）"
    return "\n".join(f"• {f}" for f in files)


def refresh_stats() -> str:
    """刷新知识库统计"""
    agent = get_agent()
    stats = agent.get_knowledge_stats()
    if stats.get("status") == "not_initialized" or stats.get("total_vectors", 0) == 0:
        return "知识库为空"
    return (
        f"📄 文件: {stats.get('files_count', 0)} 个\n"
        f"🧩 片段: {stats.get('total_vectors', 0)} 个"
    )


def refresh_sessions():
    """刷新会话列表（返回 gr.Dropdown.update 同时更新选项和选中值）"""
    agent = get_agent()
    sessions = agent.get_sessions()
    if not sessions:
        return gr.update(choices=["（无会话）"], value="（无会话）")
    choices = [
        f"{s['session_name']} ({s['session_id']})"
        for s in sessions
    ]
    return gr.update(choices=choices, value=choices[0])


def new_session() -> str:
    """创建新会话"""
    agent = get_agent()
    session_id = agent.new_session()
    return f"已创建新会话: {session_id}"


def switch_session(session_label) -> tuple:
    """切换会话，返回 (提示消息, 历史消息列表)"""
    # 防 Gradio 内部事件传递列表（demo.load 刷新 dropdown 时可能触发）
    if isinstance(session_label, list):
        session_label = session_label[-1] if session_label else None
    if not session_label or session_label == "（无会话）":
        return "请选择有效的会话", []

    # 从标签提取 session_id: "会话名 (abc123)"
    import re
    match = re.search(r'\(([a-f0-9]+)\)', session_label)
    if not match:
        return "无法解析会话 ID", []

    session_id = match.group(1)
    agent = get_agent()
    if agent.switch_session(session_id):
        # 加载该会话的历史消息
        history = agent.get_session_history(session_id)
        return f"已切换到会话: {session_id}", history
    else:
        return "切换失败，会话不存在", []


def clear_knowledge() -> str:
    """清空知识库"""
    import gr as _gr
    agent = get_agent()
    agent.clear_knowledge()
    return "🗑️ 知识库已清空"


def delete_file_and_refresh(filename: str):
    """删除指定文件并刷新界面"""
    if not filename or not filename.strip():
        return "请输入要删除的文件名", refresh_file_list(), refresh_stats()
    agent = get_agent()
    ok = agent.delete_file(filename.strip())
    if ok:
        return f"✅ 已删除: {filename.strip()}", refresh_file_list(), refresh_stats()
    else:
        return f"❌ 删除失败: {filename.strip()}（文件不存在或出错）", refresh_file_list(), refresh_stats()


# ── UI 构建 ──────────────────────────────────────

CSS = """
#app-header {
    text-align: center;
    margin-bottom: 20px;
}
#app-header h1 {
    font-size: 24px;
    margin-bottom: 4px;
}
#app-header p {
    color: #888;
    font-size: 14px;
    margin-top: 0;
}
.sidebar-section {
    margin-bottom: 20px;
}
.sidebar-section h3 {
    font-size: 14px;
    margin-bottom: 8px;
    color: #555;
    border-bottom: 1px solid #eee;
    padding-bottom: 4px;
}
"""


def build_ui():
    with gr.Blocks() as demo:
        # ── 标题 ──
        with gr.Row():
            with gr.Column():
                gr.HTML(
                    """
                    <div id="app-header">
                        <h1>📚 RAG 智能问答助手</h1>
                        <p>上传文档 → AI 自动索引 → 智能问答</p>
                    </div>
                    """
                )

        with gr.Row(equal_height=False):
            # ── 左侧栏 ──
            with gr.Column(scale=1, min_width=280):
                # 📁 文件上传
                with gr.Group(elem_classes="sidebar-section"):
                    gr.Markdown("### 📁 知识库")
                    file_input = gr.File(
                        label="上传文件",
                        file_types=[".pdf", ".docx", ".pptx", ".txt", ".md", ".csv"],
                        file_count="single",
                    )
                    upload_btn = gr.UploadButton(
                        "📤 选择文件上传",
                        file_types=[".pdf", ".docx", ".pptx", ".txt", ".md", ".csv"],
                        variant="primary",
                        file_count="single",
                    )

                # 已上传文件列表
                with gr.Group(elem_classes="sidebar-section"):
                    file_list = gr.Textbox(
                        label="已上传文件",
                        value="（暂无文件）",
                        lines=6,
                        interactive=False,
                        max_lines=10,
                    )
                    with gr.Row():
                        refresh_files_btn = gr.Button("🔄 刷新", size="sm", scale=1)
                        clear_kb_btn = gr.Button("🗑️ 清空", size="sm", scale=1, variant="stop")

                # 📊 知识库统计
                with gr.Group(elem_classes="sidebar-section"):
                    stats_box = gr.Textbox(
                        label="知识库统计",
                        value="知识库为空",
                        lines=3,
                        interactive=False,
                    )

                # 🗑️ 删除文件
                with gr.Group(elem_classes="sidebar-section"):
                    gr.Markdown("### 🗑️ 删除文件")
                    with gr.Row():
                        delete_filename = gr.Textbox(
                            label="输入要删除的文件名",
                            placeholder="如：简历.pdf",
                            scale=3,
                        )
                        delete_btn = gr.Button("删除", variant="stop", scale=1, min_width=60)

                # 📋 会话管理
                with gr.Group(elem_classes="sidebar-section"):
                    gr.Markdown("### 💬 会话")
                    session_dropdown = gr.Dropdown(
                        label="切换会话",
                        choices=["（无会话）"],
                        value="（无会话）",
                        interactive=True,
                    )
                    with gr.Row():
                        new_session_btn = gr.Button("➕ 新会话", size="sm", scale=1)
                        refresh_session_btn = gr.Button("🔄 刷新", size="sm", scale=1)

                session_status = gr.Textbox(
                    label="操作提示",
                    value="就绪",
                    lines=2,
                    interactive=False,
                )

            # ── 右侧对话区 ──
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="对话",
                    height=500,
                    placeholder="上传文档后，在这里提问...",
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="输入问题",
                        placeholder="输入你的问题，按 Enter 发送...",
                        scale=8,
                        container=False,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1, min_width=80)
                clear_chat_btn = gr.Button("🗑️ 清空对话", size="sm")

        # ── 事件绑定 ──────────────────────────────

        # 对话处理（流式）
        def respond(message, history):
            if not message or not message.strip():
                yield "", history
                return
            agent = get_agent()
            session_id = agent.get_current_session_id()
            if session_id is None:
                session_id = agent.new_session()

            history.append({"role": "user", "content": message.strip()})

            full_response = ""
            for partial in agent.chat_stream(message.strip(), session_id):
                full_response = partial
                new_history = history + [{"role": "assistant", "content": full_response}]
                yield "", new_history

            yield "", history + [{"role": "assistant", "content": full_response}]

        msg_input.submit(respond, [msg_input, chatbot], [msg_input, chatbot])
        send_btn.click(respond, [msg_input, chatbot], [msg_input, chatbot])

        # 清空对话
        clear_chat_btn.click(lambda: None, None, chatbot, queue=False)

        # 文件上传
        upload_btn.upload(
            fn=lambda f: (upload_file(f), refresh_file_list(), refresh_stats()),
            inputs=[upload_btn],
            outputs=[session_status, file_list, stats_box],
        )

        # 刷新文件列表
        refresh_files_btn.click(
            fn=refresh_file_list,
            outputs=[file_list],
        )
        refresh_files_btn.click(
            fn=refresh_stats,
            outputs=[stats_box],
        )

        # 清空知识库
        clear_kb_btn.click(
            fn=lambda: (
                clear_knowledge(),
                refresh_file_list(),
                refresh_stats(),
            ),
            outputs=[session_status, file_list, stats_box],
        )

        # 删除文件
        delete_btn.click(
            fn=delete_file_and_refresh,
            inputs=[delete_filename],
            outputs=[session_status, file_list, stats_box],
        )

        # 刷新会话（加载首个会话历史到聊天框）
        def handle_refresh_sessions():
            session_update = refresh_sessions()
            current_label = getattr(session_update, 'value', None)
            if current_label and current_label != "（无会话）":
                _, history = switch_session(current_label)
            else:
                history = []
            return session_update, history

        refresh_session_btn.click(
            fn=handle_refresh_sessions,
            outputs=[session_dropdown, chatbot],
        )

        # 新会话（清空聊天框）
        new_session_btn.click(
            fn=lambda: (new_session(), refresh_sessions(), []),
            outputs=[session_status, session_dropdown, chatbot],
        )

        # 切换会话（加载历史到聊天框）
        session_dropdown.change(
            fn=switch_session,
            inputs=[session_dropdown],
            outputs=[session_status, chatbot],
        )

        # 页面加载时刷新（加载会话历史到聊天框）
        demo.load(
            fn=initial_load,
            outputs=[file_list, stats_box, session_dropdown, chatbot],
        )

    # 启用队列以支持流式输出
    demo.queue(default_concurrency_limit=5)
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_port=config.GRADIO_SERVER_PORT,
        share=config.GRADIO_SHARE,
        inbrowser=True,
        css=CSS,
    )
