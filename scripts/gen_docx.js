const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat, HeadingLevel,
  BorderStyle, WidthType, ShadingType, VerticalAlign, PageNumber,
  PageBreak, ExternalHyperlink,
} = require("docx");

// ============================================================
// Helper functions
// ============================================================

const FONT = "Arial";
const ACCENT = "1F4E79"; // dark blue
const MARGIN = 1440; // 1 inch

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const cellBorders = { top: border, bottom: border, left: border, right: border };

function headerCell(text, width) {
  return new TableCell({
    borders: cellBorders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: "1F4E79", type: ShadingType.CLEAR },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 60, after: 60 },
      children: [new TextRun({ text, bold: true, color: "FFFFFF", font: FONT, size: 20 })],
    })],
  });
}

function dataCell(text, width, opts = {}) {
  return new TableCell({
    borders: cellBorders,
    width: { size: width, type: WidthType.DXA },
    verticalAlign: VerticalAlign.CENTER,
    shading: opts.shading ? { fill: opts.shading, type: ShadingType.CLEAR } : undefined,
    children: [new Paragraph({
      spacing: { before: 40, after: 40 },
      children: [new TextRun({ text: String(text), font: FONT, size: 20, ...opts.run })],
    })],
  });
}

function heading(level, text) {
  return new Paragraph({
    heading: level,
    spacing: { before: level === HeadingLevel.HEADING_1 ? 360 : 240, after: 120 },
    children: [new TextRun({ text, font: FONT })],
  });
}

function para(text, opts = {}) {
  return new Paragraph({
    spacing: { after: opts.afterSpacing ?? 120, before: opts.beforeSpacing ?? 0 },
    alignment: opts.alignment,
    children: [new TextRun({ text, font: FONT, size: 22, ...opts.run })],
  });
}

function boldPara(label, value) {
  return new Paragraph({
    spacing: { after: 80 },
    children: [
      new TextRun({ text: label, font: FONT, size: 22, bold: true }),
      new TextRun({ text: value, font: FONT, size: 22 }),
    ],
  });
}

function bulletItem(text, ref = "bullet-list", level = 0) {
  return new Paragraph({
    numbering: { reference: ref, level },
    spacing: { after: 60 },
    children: [new TextRun({ text, font: FONT, size: 22 })],
  });
}

function codeBlock(lines) {
  return lines.map(line =>
    new Paragraph({
      spacing: { after: 0, before: 0 },
      indent: { left: 480 },
      children: [new TextRun({ text: line, font: "Consolas", size: 18, color: "333333" })],
    })
  );
}

function emptyLine() {
  return new Paragraph({ spacing: { after: 60 }, children: [] });
}

// ============================================================
// Bullet list config
// ============================================================
const numbering = {
  config: [
    {
      reference: "bullet-list",
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } },
      }],
    },
    {
      reference: "sub-bullet",
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: "◦", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 1080, hanging: 360 } } },
      }],
    },
  ],
};

// ============================================================
// Section content builders
// ============================================================

function buildTechStack() {
  const W1 = 2200, W2 = 7160;
  const rows = [
    ["LLM", "DeepSeek Chat (deepseek-chat)，通过 OpenAI SDK 协议调用，支持流式/非流式，自动指数退避重试"],
    ["Embedding", "sentence-transformers (384d) paraphrase-multilingual-MiniLM-L12-v2，多语言语义模型；备选 LocalEmbedding（256d 纯 numpy，零依赖）"],
    ["向量数据库", "ChromaDB PersistentClient，本地持久化，cosine 距离，增量索引"],
    ["Agent 框架", "LlamaIndex ReAct Agent，自定义 FixedReActAgentWorker 修复 DeepSeek 流式截断问题"],
    ["后端框架", "FastAPI + Uvicorn，12 个 RESTful API + SSE 流式端点，异步非阻塞"],
    ["前端", "原生 SPA (HTML/CSS/JS)，三栏布局，暗色主题，SSE 流式逐字渲染，marked.js 本地化"],
    ["持久化", "SQLite，对话历史持久化 + 自动摘要压缩（4000 token 阈值）"],
    ["评估", "RAGAS，4 项指标：faithfulness / answer_relevancy / context_precision / context_recall"],
    ["文档解析", "pymupdf4llm / python-docx / python-pptx，支持 PDF / DOCX / PPTX / TXT / MD / CSV"],
  ];
  const children = [heading(HeadingLevel.HEADING_1, "一、技术栈")];

  const tableRows = [new TableRow({ tableHeader: true, children: [headerCell("层级", W1), headerCell("技术说明", W2)] })];
  rows.forEach((r, i) => {
    tableRows.push(new TableRow({
      children: [
        dataCell(r[0], W1, { run: { bold: true }, shading: i % 2 === 0 ? "F2F7FB" : undefined }),
        dataCell(r[1], W2, { shading: i % 2 === 0 ? "F2F7FB" : undefined }),
      ],
    }));
  });
  children.push(
    new Table({ columnWidths: [W1, W2], rows: tableRows }),
    emptyLine()
  );
  return children;
}

function buildArchitecture() {
  const children = [
    heading(HeadingLevel.HEADING_1, "二、四层架构"),
    emptyLine(),
  ];
  // Architecture layers
  const layers = [
    { name: "API 层", desc: "FastAPI server.py — 12 REST 端点 + SSE 流式 + 静态文件服务", color: "E8F0FE" },
    { name: "Agent 层", desc: "rag_agent.py — ReAct Agent (Thought→Action→Observation→Answer)，工具注册，对话分发", color: "E6F4EA" },
    { name: "LLM 层 + 知识层", desc: "DeepSeekLLM 自定义封装 + 文档解析/向量索引/检索增强管线（查询分解→混合检索→重排序）", color: "FEF7E0" },
    { name: "持久化层", desc: "SQLite 存储会话/消息 + 自动摘要压缩 + file_registry.json 文件去重", color: "FCE8E6" },
  ];

  const W1_arch = 1800, W2_arch = 7560;
  const archRows = [
    new TableRow({
      tableHeader: true,
      children: [headerCell("层级", W1_arch), headerCell("说明", W2_arch)],
    }),
  ];
  layers.forEach((l, i) => {
    archRows.push(new TableRow({
      children: [
        dataCell(l.name, W1_arch, { run: { bold: true, color: "1F4E79" }, shading: l.color }),
        dataCell(l.desc, W2_arch, { shading: l.color }),
      ],
    }));
  });
  children.push(
    new Table({ columnWidths: [W1_arch, W2_arch], rows: archRows }),
    emptyLine()
  );
  return children;
}

function buildRunningFlow() {
  const children = [
    heading(HeadingLevel.HEADING_1, "三、运行流程"),
    heading(HeadingLevel.HEADING_2, "启动流程"),
    para("用户启动 server.py → FastAPI 应用初始化 → lifespan 事件触发 Agent 预热："),
    bulletItem("DeepSeekLLM 初始化（OpenAI SDK + custom httpx 客户端，certifi 自动发现 SSL 证书）"),
    bulletItem("Settings.embed_model 初始化（语义模型 384d 或本地模型 256d）"),
    bulletItem("ChromaDB PersistentClient 连接 / 创建 → VectorStoreIndex 加载 / 创建索引"),
    bulletItem("PersistentChatMemory SQLite 连接 → ReAct AgentRunner 创建（注册工具）"),
    bulletItem("LLM 连接健康检测 → 浏览器自动打开 http://localhost:7860"),
    emptyLine(),
    heading(HeadingLevel.HEADING_2, "用户交互流程"),
    boldPara("上传文件: ", "浏览器拖拽文件 → POST /api/files/upload → loader.load_document() → SentenceSplitter 切分 → SemanticEmbedding 向量化 → ChromaDB insert_nodes() → file_registry 记录"),
    emptyLine(),
    boldPara("提问: ", "浏览器输入消息 → POST /api/chat/stream (SSE) → agent.chat_stream() → PersistentChatMemory 加载/压缩历史 → ReAct Agent 推理循环（工具调用 / LLM 生成）→ SSE 逐 token 推送 → 前端 marked.js 实时渲染 → 回答保存到 SQLite"),
    emptyLine(),
  ];
  return children;
}

function buildDataFlow() {
  const children = [
    heading(HeadingLevel.HEADING_1, "四、数据流"),
    heading(HeadingLevel.HEADING_2, "文件上传数据流"),
  ];
  const uploadSteps = [
    "PDF/DOCX/TXT → loader.load_document() → List[Document]",
    "SentenceSplitter(chunk_size=512, overlap=128) → 按 token 长度切分",
    "SemanticEmbedding.encode(nodes) → 384 维向量",
    "ChromaDB (knowledge_base collection, cosine 距离)",
    "file_registry.json 记录文件名（去重）",
  ];
  uploadSteps.forEach(s => bulletItem(s));

  children.push(emptyLine(), heading(HeadingLevel.HEADING_2, "问答数据流"));

  const qaSteps = [
    "用户输入 → PersistentChatMemory（SQLite 加载历史，超 4000 token 自动摘要压缩）",
    "ReAct Agent：Thought → 判断是否需要检索知识库",
    "knowledge_base 工具调用：",
    "  ① QueryDecomposer：复杂问题拆分为子查询",
    "  ② HybridRetriever：向量检索 (top-16) + BM25 (top-16) → RRF 融合 → top-8",
    "  ③ Reranker：关键词评分 (0.3) + LLM 相关性评分 (0.7) → top-5 精排",
    "DeepSeek LLM 流式生成 → SSE 推送 → 前端逐字渲染",
  ];
  qaSteps.forEach(s => bulletItem(s));
  children.push(emptyLine());
  return children;
}

function buildAgentKnowledge() {
  const children = [
    heading(HeadingLevel.HEADING_1, "五、Agent 相关知识"),
    heading(HeadingLevel.HEADING_2, "5.1 ReAct Agent 原理"),
    para("ReAct（Reasoning + Acting）是一种结合推理和行动的 Agent 范式。模型在每步推理中交替执行："),
    bulletItem("Thought：分析当前状态，决定下一步做什么"),
    bulletItem("Action：调用工具（检索知识库 / 查统计）"),
    bulletItem("Observation：工具返回的结果"),
    bulletItem("Answer：收集足够信息后给出最终答案"),
    emptyLine(),
    heading(HeadingLevel.HEADING_2, "5.2 FixedReActAgentWorker（自定义修复）"),
    para("DeepSeek 的流式输出行为与 OpenAI 不同——有时输出不以 Thought: 开头。原版 ReActAgentWorker._infer_stream_chunk_is_final 会误判首个 chunk 为最终答案，提前终止 ReAct 循环。"),
    para("修复方案：继承 ReActAgentWorker 重写该方法，只有确认 chunk 包含 Answer: 时才视为推理结束，否则继续收集。"),
    emptyLine(),
    heading(HeadingLevel.HEADING_2, "5.3 检索增强管线（Phase 3）"),
  ];
  const W1_p = 2400, W2_p = 6960;
  const phaseRows = [
    new TableRow({
      tableHeader: true,
      children: [headerCell("阶段", W1_p), headerCell("说明", W2_p)],
    }),
    new TableRow({
      children: [
        dataCell("QueryDecomposer", W1_p, { run: { bold: true }, shading: "F2F7FB" }),
        dataCell("LLM 判断是否为复杂查询 → 拆分为 2-3 子查询，分别检索后合并", W2_p, { shading: "F2F7FB" }),
      ],
    }),
    new TableRow({
      children: [
        dataCell("HybridRetriever", W1_p, { run: { bold: true } }),
        dataCell("向量检索 (semantic) top-16 + BM25 关键词检索 top-16 → RRF 融合 top-8", W2_p),
      ],
    }),
    new TableRow({
      children: [
        dataCell("Reranker", W1_p, { run: { bold: true }, shading: "F2F7FB" }),
        dataCell("关键词评分 (0.3) + LLM 语义评分 (0.7) → 加权融合 top-5", W2_p, { shading: "F2F7FB" }),
      ],
    }),
  ];
  children.push(
    new Table({ columnWidths: [W1_p, W2_p], rows: phaseRows }),
    emptyLine()
  );

  // Memory system
  children.push(heading(HeadingLevel.HEADING_2, "5.4 记忆系统"));
  const memW1 = 1600, memW2 = 1600, memW3 = 2200, memW4 = 3960;
  const memRows = [
    new TableRow({
      tableHeader: true,
      children: [headerCell("层级", memW1), headerCell("存储", memW2), headerCell("作用", memW3), headerCell("机制", memW4)],
    }),
    new TableRow({
      children: [
        dataCell("短期", memW1, { run: { bold: true }, shading: "F2F7FB" }),
        dataCell("ChatMemoryBuffer", memW2, { shading: "F2F7FB" }),
        dataCell("Agent 工作记忆", memW3, { shading: "F2F7FB" }),
        dataCell("LlamaIndex 内置对话上下文", memW4, { shading: "F2F7FB" }),
      ],
    }),
    new TableRow({
      children: [
        dataCell("长期", memW1, { run: { bold: true } }),
        dataCell("SQLite", memW2),
        dataCell("跨会话持久化", memW3),
        dataCell("每次对话保存到 conversations.db", memW4),
      ],
    }),
    new TableRow({
      children: [
        dataCell("压缩", memW1, { run: { bold: true }, shading: "F2F7FB" }),
        dataCell("LLM 摘要", memW2, { shading: "F2F7FB" }),
        dataCell("防超长上下文", memW3, { shading: "F2F7FB" }),
        dataCell("超 4000 token 自动压缩历史为摘要", memW4, { shading: "F2F7FB" }),
      ],
    }),
  ];
  children.push(
    new Table({ columnWidths: [memW1, memW2, memW3, memW4], rows: memRows }),
    emptyLine(),
    heading(HeadingLevel.HEADING_2, "5.5 安全与容错"),
    bulletItem("指数退避重试：API 限速 (429/503) 时自动重试，最长等待 8s"),
    bulletItem("Agent 自愈重试：第一次流式调用失败 → 重置 Agent 和记忆 → 自动重试"),
    bulletItem("API Key 三级加载：环境变量 → .env → Fernet+PBKDF2 加密文件（600K 迭代）"),
    bulletItem("SSL 证书：使用 certifi 自动定位，不依赖系统环境变量"),
    bulletItem("代理绕过：DeepSeek/OpenAI API 域名自动加入 NO_PROXY，不受系统代理干扰"),
    emptyLine()
  );
  return children;
}

function buildHighlights() {
  const children = [heading(HeadingLevel.HEADING_1, "六、项目亮点（简历要点）")];

  // Technical highlights
  children.push(heading(HeadingLevel.HEADING_2, "技术亮点"));

  const highlights = [
    {
      title: "端到端 RAG 系统自研",
      desc: "从文档解析 → 向量化 → 检索增强 → LLM 生成 → 前端展示，全链路自实现。采用 FastAPI + 原生 SPA，生产可用。",
    },
    {
      title: "检索增强管线深度优化（Phase 3）",
      desc: "查询分解器解决跨主题复杂问题；混合检索融合语义 + 关键词，弥补纯向量对精确术语匹配的不足；LLM 两阶段重排序提升精准度。",
    },
    {
      title: "Custom LLM 封装",
      desc: "绕过 LlamaIndex OpenAI 模型名校验，自定义 DeepSeek LLM，支持任意 OpenAI 兼容 API（DeepSeek/Qwen/GLM 等）。内置指数退避重试、SSL 自动发现、代理绕过。",
    },
    {
      title: "ReAct Agent 流式修复",
      desc: "发现并修复了 DeepSeek × LlamaIndex 的流式推理截断 bug，确保长程多步推理中 Agent 不会过早终止。",
    },
    {
      title: "SSE 流式架构",
      desc: "FastAPI StreamingResponse + 前端 ReadableStream，逐 token 推送，marked.js 实时渲染，用户体验接近 ChatGPT。",
    },
    {
      title: "双 Embedding 策略",
      desc: "SemanticEmbedding (384d) 理解语义同义词；LocalEmbedding (256d) 零依赖快速启动。通过配置一键切换。",
    },
    {
      title: "API Key 安全体系",
      desc: "三级加载：环境变量 → .env → Fernet+PBKDF2 加密文件（600K 迭代），加密文件可提交仓库。",
    },
  ];

  highlights.forEach(h => {
    children.push(
      new Paragraph({
        spacing: { after: 40 },
        children: [
          new TextRun({ text: h.title, font: FONT, size: 22, bold: true, color: "1F4E79" }),
        ],
      }),
      para(h.desc, { afterSpacing: 80 })
    );
  });

  // Engineering highlights
  children.push(heading(HeadingLevel.HEADING_2, "工程亮点"));
  const engItems = [
    "全类型文档解析：PDF（含表格/图片）、DOCX、PPTX、TXT、MD、CSV",
    "增量索引：文件注册表去重，避免重复索引",
    "持久化记忆：SQLite 存储对话历史，4000 token 自动摘要压缩",
    "会话管理：多会话支持，历史回溯，删除会话",
    "RAGAS 质量评估：faithfulness / relevancy / precision / recall 四项指标",
    "响应式前端：三栏布局，移动端自适应，暗色主题，拖拽上传",
  ];
  engItems.forEach(e => bulletItem(e));
  children.push(emptyLine());
  return children;
}

function buildEvaluation() {
  const children = [heading(HeadingLevel.HEADING_1, "七、评估体系")];
  para("使用 RAGAS 框架对系统进行量化评估，在独立 ChromaDB 环境中运行，不污染生产数据。4 篇测试文档 + 10 个问答对，覆盖单文档事实查询、跨文档综合、多跳推理等场景。");
  emptyLine();

  const W1_e = 2400, W2_e = 2800, W3_e = 4160;
  const evalRows = [
    new TableRow({
      tableHeader: true,
      children: [headerCell("指标", W1_e), headerCell("说明", W2_e), headerCell("测量内容", W3_e)],
    }),
    new TableRow({
      children: [
        dataCell("faithfulness", W1_e, { run: { bold: true }, shading: "F2F7FB" }),
        dataCell("忠实度", W2_e, { shading: "F2F7FB" }),
        dataCell("生成答案是否基于检索到的上下文，有无幻觉", W3_e, { shading: "F2F7FB" }),
      ],
    }),
    new TableRow({
      children: [
        dataCell("answer_relevancy", W1_e, { run: { bold: true } }),
        dataCell("答案相关性", W2_e),
        dataCell("答案与问题的相关程度", W3_e),
      ],
    }),
    new TableRow({
      children: [
        dataCell("context_precision", W1_e, { run: { bold: true }, shading: "F2F7FB" }),
        dataCell("上下文精度", W2_e, { shading: "F2F7FB" }),
        dataCell("检索到的片段中，有多少是真正相关的", W3_e, { shading: "F2F7FB" }),
      ],
    }),
    new TableRow({
      children: [
        dataCell("context_recall", W1_e, { run: { bold: true } }),
        dataCell("上下文召回率", W2_e),
        dataCell("所有相关片段中，检索到了多少", W3_e),
      ],
    }),
  ];
  children.push(
    new Table({ columnWidths: [W1_e, W2_e, W3_e], rows: evalRows }),
    emptyLine(),
    para("评估命令:", { run: { bold: true } }),
    ...codeBlock([
      "conda activate all-in-rag",
      "python evaluation/run_eval.py          # 运行评估",
      "python evaluation/run_eval.py --keep   # 保留临时数据",
    ]),
    emptyLine()
  );
  return children;
}

function buildQuickStart() {
  const children = [heading(HeadingLevel.HEADING_1, "八、快速启动")];
  const steps = [
    "安装依赖：conda activate all-in-rag && python -m pip install -r requirements.txt",
    "配置 API Key：export DEEPSEEK_API_KEY=\"sk-xxxxx\"",
    "启动服务：python server.py（浏览器自动打开 http://localhost:7860）",
    "可选：下载语义模型（国内网络需走镜像）：python scripts/download_model.py",
  ];
  steps.forEach(s => bulletItem(s));
  children.push(emptyLine());
  return children;
}

function buildProjectStructure() {
  const children = [heading(HeadingLevel.HEADING_1, "九、项目目录结构")];
  const tree = [
    "RAG-question/",
    "├── server.py                    # FastAPI 入口（12 API + SSE + 静态文件）",
    "├── config.py                    # 全局配置 + API Key 三级加载",
    "├── agent/                       # Agent 核心层",
    "│   ├── rag_agent.py             # Agent 中枢：工具/记忆/对话",
    "│   ├── deepseek_llm.py          # DeepSeek LLM 自定义封装",
    "│   ├── llm_adapter.py           # LLM 配置工厂",
    "│   └── memory.py                # SQLite 持久化记忆 + 摘要压缩",
    "├── knowledge/                   # 知识层（检索增强管线）",
    "│   ├── loader.py                # 多格式文档解析",
    "│   ├── indexer.py               # ChromaDB 索引管理",
    "│   ├── semantic_embedding.py    # 语义 Embedding (384d)",
    "│   ├── local_embedding.py       # 本地 Embedding (256d, 零下载)",
    "│   ├── hybrid_retriever.py      # 向量 + BM25 混合检索",
    "│   ├── reranker.py              # 两阶段重排序",
    "│   └── query_decomposer.py      # LLM 查询分解",
    "├── evaluation/                  # RAGAS 评估",
    "├── static/                      # 前端 SPA（index.html + marked.min.js）",
    "├── docs/                        # 项目文档",
    "├── tests/                       # 单元测试",
    "└── scripts/                     # 工具脚本",
  ];
  tree.forEach(line => {
    const isDir = line.endsWith("/") || line.match(/^[│└├]/);
    children.push(new Paragraph({
      spacing: { after: 0, before: 0 },
      indent: { left: 240 },
      children: [new TextRun({
        text: line,
        font: "Consolas",
        size: 18,
        color: isDir && line.includes("/") ? "1F4E79" : "333333",
        bold: isDir && line.includes("/"),
      })],
    }));
  });
  children.push(emptyLine());
  return children;
}

// ============================================================
// Build complete document
// ============================================================

const children = [
  // Title page
  new Paragraph({ spacing: { before: 3000 }, children: [] }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
    children: [new TextRun({ text: "RAG 智能问答 Agent", font: FONT, size: 52, bold: true, color: "1F4E79" })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 600 },
    children: [new TextRun({ text: "项目文档", font: FONT, size: 36, color: "666666" })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
    children: [new TextRun({ text: "基于 LlamaIndex + DeepSeek + ChromaDB 的检索增强生成系统", font: FONT, size: 24, color: "888888", italics: true })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 100 },
    children: [new TextRun({ text: "FastAPI 后端  |  原生 SPA 前端  |  SSE 流式  |  ReAct Agent", font: FONT, size: 22, color: "888888" })],
  }),
  new Paragraph({ children: [new PageBreak()] }),

  // TOC
  heading(HeadingLevel.HEADING_1, "目录"),
  para("一、技术栈"),
  para("二、四层架构"),
  para("三、运行流程"),
  para("四、数据流"),
  para("五、Agent 相关知识"),
  para("六、项目亮点（简历要点）"),
  para("七、评估体系"),
  para("八、快速启动"),
  para("九、项目目录结构"),
  new Paragraph({ children: [new PageBreak()] }),

  // Content sections
  ...buildTechStack(),
  ...buildArchitecture(),
  ...buildRunningFlow(),
  ...buildDataFlow(),
  ...buildAgentKnowledge(),
  ...buildHighlights(),
  ...buildEvaluation(),
  ...buildQuickStart(),
  ...buildProjectStructure(),

  // Footer note
  emptyLine(),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 600 },
    children: [new TextRun({ text: "— 文档生成日期 2026-06-24 —", font: FONT, size: 18, color: "999999", italics: true })],
  }),
];

// ============================================================
// Document assembly
// ============================================================

const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, color: "1F4E79", font: FONT },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 },
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, color: "2E75B6", font: FONT },
        paragraph: { spacing: { before: 240, after: 160 }, outlineLevel: 1 },
      },
    ],
  },
  numbering,
  sections: [{
    properties: {
      page: {
        margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN },
        pageNumbers: { start: 1 },
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: "RAG 智能问答 Agent — 项目文档", font: FONT, size: 16, color: "999999", italics: true })],
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "第 ", font: FONT, size: 18, color: "999999" }),
            new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18, color: "999999" }),
            new TextRun({ text: " 页", font: FONT, size: 18, color: "999999" }),
          ],
        })],
      }),
    },
    children,
  }],
});

const outPath = "C:/Users/86182/Desktop/RAG-question/docs/project_overview.docx";
Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(outPath, buffer);
  console.log("✅ Word 文档已生成:", outPath);
});
