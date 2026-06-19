 # 掌柜智库 (Zhiku RAG)
 
 ## 1. 项目概述
 
 掌柜智库是一个基于 **RAG（检索增强生成）** 架构的智能知识库问答系统，专注于对产品技术文档（如产品手册、安全指南、使用说明书等 PDF/Markdown 文件）进行自动化导入、向量化存储，并支持多轮对话式的智能查询问答。
 
 项目采用 **LangGraph** 工作流引擎，将知识库构建（导入链路）和智能问答（查询链路）组织为两个独立的有向图流程，模块清晰、扩展性强。
 
 ---
 
 ## 2. 系统架构
 
 两个独立的 LangGraph 工作流构成系统核心：
 
 **导入流程：** 上传文档 → PDF/MD 解析 → 图片处理 → 文档切分 → 商品名识别 → BGE 向量化 → 写入 Milvus
 
 **查询流程：** 用户问题 → 商品名确认 → 多路检索（稠密向量 + HyDE + 联网搜索）→ RRF 融合 → BGE 重排序 → LLM 生成答案（支持流式/同步）
 
 两套 FastAPI 服务分别承载导入和查询功能，共享底层数据基础设施（Milvus / MongoDB / MinIO / Neo4j）。
 
 ---
 
 ## 3. 技术栈
 
 | 类别 | 技术 | 用途 |
 |------|------|------|
 | **Web 框架** | FastAPI + Uvicorn | 提供 REST API 和 SSE 流式接口 |
 | **工作流引擎** | LangGraph | 编排导入/查询流程的有向图节点 |
 | **大语言模型** | 阿里云灵积 DashScope (Qwen-Flash / Qwen3-VL-Flash) | 商品名识别、查询改写、答案生成 |
 | **向量嵌入** | BGE-M3 (BAAI/bge-m3) | 稠密向量 + 稀疏向量混合编码 |
 | **重排序** | BGE-Reranker-Large | Cross-Encoder 精排打分 |
 | **向量数据库** | Milvus | 存储文档切片的稠密/稀疏向量，支持混合搜索 |
 | **图数据库** | Neo4j | 预留的知识图谱存储 |
 | **文档存储** | MinIO (S3 兼容) | 图片上传与访问 |
 | **对话历史** | MongoDB | 存储多轮对话历史记录 |
 | **PDF 解析** | MinerU API | PDF 转 Markdown |
 | **联网搜索** | 阿里云百炼 MCP (WebSearch) | 通过 MCP 协议调用外部搜索 |
 | **运行环境** | Python >= 3.11, PyTorch, uv | |
 
 ---
 
 ## 4. 导入流程 (Import Pipeline)
 
 导入流程将产品文档（PDF 或 Markdown）经过一系列处理存入向量数据库，用于后续检索。
 
 **流程节点：**
 
 1. **node_entry** — 入口节点：校验文件路径，判定文件类型（PDF/MD），提取文件标题
 2. **node_pdf_to_md** — PDF 转 Markdown：调用 MinerU API 上传 PDF 并轮询解析结果，下载 Markdown 文件
 3. **node_md_img** — 图片处理：提取 Markdown 中的图片链接，下载图片并上传至 MinIO，用 VL 模型生成图片描述摘要替换原图引用
 4. **node_document_split** — 文档切分：按标题层级（Markdown 标题）进行语义切分，对过短的块进行邻近合并，确保每个切块语义完整
 5. **node_item_name_recognition** — 商品名识别：调用 LLM 从文档前几个切片中识别商品名称（如"华为 Mate60 Pro"），并为该名称生成稠密/稀疏向量存入 Milvus item_name 集合
 6. **node_bge_embedding** — 向量化：使用 BGE-M3 模型批量生成每个切片的稠密向量 (dense) 和稀疏向量 (sparse)
 7. **node_import_milvus** — 入库：将带向量的切片数据写入 Milvus `kb_chunks` 集合
 
 **API 入口：**
 - `GET /import` — 返回导入页面
 - `POST /upload` — 上传文件，启动后台导入任务
 - `GET /status/{task_id}` — 查询导入任务状态
 
 ---
 
 ## 5. 查询流程 (Query Pipeline)
 
 查询流程接收用户提问，执行多路检索和融合排序，最终生成答案。
 
 **流程节点：**
 
 1. **node_item_name_confirm** — 商品名确认：从 MongoDB 加载最近对话历史，调用 LLM 提取用户问题中的商品名并重写查询，在 Milvus `kb_item_names` 集合中进行向量相似度搜索匹配
 2. **node_search_embedding** — 稠密/稀疏向量检索：将重写的查询编码为 BGE-M3 向量，在 Milvus 中进行混合搜索
 3. **node_search_embedding_hyde** — HyDE 检索：先生成一个假设性回答（HyDE），再将假设回答编码为向量进行检索，提升召回
 4. **node_web_search_mcp** — 联网搜索：通过阿里云百炼 MCP 协议的 WebSearch 工具进行外部搜索，补充知识库之外的实时信息
 5. **node_rrf** — 倒排融合排序 (Reciprocal Rank Fusion)：将多路检索结果按 RRF 算法加权重排融合
 6. **node_rerank** — BGE 重排序：将 RRF 融合结果与联网搜索结果合并，使用 BGE-Reranker-Large 模型进行 Cross-Encoder 精确打分，并基于分数断层进行动态 Top-K 截断
 7. **node_answer_output** — 答案生成与输出：组装上下文（检索文档 + 历史对话 + 商品名），调用 LLM 生成答案，从文档中提取图片 URL 一并返回，将本次对话保存到 MongoDB
 
 **API 入口：**
 - `GET /chat.html` — 返回聊天页面
 - `POST /query` — 发送查询请求（支持流式/同步）
 - `GET /stream/{session_id}` — SSE 长连接流式接收答案
 - `GET /history/{session_id}` — 查看对话历史
 - `DELETE /history/{session_id}` — 清空对话历史
 - `GET /health` — 健康检查
 
 在 node_item_name_confirm 中，根据向量匹配分数（>=0.85 确认、>=0.6 候选）决定走检索还是让用户确认商品名。
 
 ---
 
 ## 6. Prompt 设计
 
 所有提示词模板位于 `prompts/` 目录：
 
 | 文件名 | 用途 |
 |--------|------|
 | `answer_out.prompt` | 最终答案生成：基于参考内容 + 历史 + 商品名回答问题，支持图片 URL 提取 |
 | `hyde_prompt.prompt` | HyDE 检索用的假设性回答生成 |
 | `item_name_recognition.prompt` | 文档导入时从文本切片识别商品名称 |
 | `product_recognition_system.prompt` | 商品识别系统的 System Prompt |
 | `rewritten_query_and_itemnames.prompt` | 查询时根据历史对话改写用户问题并提取商品名 |
 | `image_summary.prompt` | 图片描述摘要生成（VL 模型） |
 
 ---
 
 ## 7. 环境依赖
 
 ### 核心模型
 
 | 模型 | 来源 | 用途 |
 |------|------|------|
 | BAAI/bge-m3 | ModelScope / HuggingFace | 文本向量编码（稠密 + 稀疏） |
 | BAAI/bge-reranker-large | ModelScope / HuggingFace | 检索结果重排序 |
 
 模型缓存路径通过 `.env` 文件配置（`BGE_M3_PATH`、`BGE_RERANKER_LARGE` 等）。
 
 ### 外部服务
 
 | 服务 | 配置项 | 说明 |
 |------|--------|------|
 | 阿里云百炼 DashScope | `OPENAI_API_KEY`, `OPENAI_BASE_URL` | LLM 调用和 VL 模型 |
 | Milvus | `MILVUS_URL` | 向量数据库（默认 localhost:19530） |
 | MongoDB | `MONGO_URL` | 对话历史存储（默认 localhost:27017） |
 | Neo4j | `NEO4J_URI` | 图数据库（预留，默认 localhost:7687） |
 | MinIO | `MINIO_ENDPOINT` | 图片对象存储（默认 localhost:9000） |
 | MinerU | `MINERU_BASE_URL`, `MINERU_API_TOKEN` | PDF 解析服务 |
 | 百炼 MCP | `MCP_DASHSCOPE_BASE_URL` | 联网搜索接口 |
 
 ---
 
 ## 8. 项目目录结构
 
 ```
 zhiku_rag/
 ├── app/
 │   ├── conf/                 # 配置类（LM、Milvus、MinIO、MinerU、Embedding、Reranker）
 │   ├── core/                 # 核心工具（日志 Logger、Prompt 加载）
 │   ├── lm/                   # 模型工具（LLM 客户端、BGE-M3 嵌入、BGE-Reranker）
 │   ├── clients/              # 数据层客户端（Milvus、MinIO、MongoDB、Neo4j）
 │   ├── utils/                # 工具函数（SSE、速率限制、任务追踪、字符串转义等）
 │   ├── tool/                 # 模型下载工具
 │   ├── import_process/       # 导入流程
 │   │   ├── agent/            # LangGraph 工作流（main_graph.py、state.py）
 │   │   │   └── nodes/        # 7 个节点文件
 │   │   ├── api/              # FastAPI 路由
 │   │   └── page/             # 前端页面 (import.html)
 │   ├── query_process/        # 查询流程
 │   │   ├── agent/            # LangGraph 工作流（main_graph.py、state.py）
 │   │   │   └── nodes/        # 7 个节点文件
 │   │   ├── api/              # FastAPI 路由
 │   │   ├── page/             # 前端页面 (chat.html, query_monitor.html)
 │   │   └── sse/              # SSE 流式示例页面
 │   └── test/                 # 测试文件
 ├── prompts/                  # 提示词模板 (.prompt)
 ├── logs/                     # 运行日志（按日期自动轮转）
 ├── output/                   # 导入处理输出（按日期/任务 ID 分目录）
 ├── .env                      # 环境变量配置
 ├── pyproject.toml            # 项目元数据与依赖
 ├── requirements.txt          # 依赖清单
 └── uv.lock                   # uv 锁定文件
 ```
 
 ---
 
 ## 9. 快速开始
 
 ### 环境准备
 
 1. **安装 Python 3.11+ 与 uv**
 
 2. **克隆项目并安装依赖**
 
    ```bash
    cd zhiku_rag
    uv sync
    ```
 
 3. **配置环境变量**
    复制 `.env` 文件并填写：
    - 百炼 DashScope API Key（`OPENAI_API_KEY`）
    - Milvus / MongoDB / Neo4j / MinIO 连接地址
    - MinerU API Token（`MINERU_API_TOKEN`）
    - BGE 模型本地路径
 
 4. **下载模型**
 
    ```bash
    python app/tool/download_bgem3.py
    python app/tool/download_reranker.py
    ```
 
 5. **启动外部服务**
    - 启动 Milvus（`milvus-server`）
    - 启动 MongoDB
    - 启动 MinIO
    - （可选）启动 Neo4j
 
 ### 启动服务
 
 ```bash
 # 启动导入服务（端口 8081）
 uvicorn app.import_process.api.import_server:app --host 127.0.0.1 --port 8081
 
 # 启动查询服务（端口 8082）
 uvicorn app.query_process.api.query_server:app --host 127.0.0.1 --port 8082
 ```
 
 ### 使用流程
 
 1. 访问 `http://127.0.0.1:8081/import` 上传产品文档
 2. 系统自动完成文档解析 -> 切分 -> 向量化 -> 入库
 3. 访问 `http://127.0.0.1:8082/chat.html` 进行智能问答
 
 ---
 
 ## 10. 设计亮点
 
 - **混合检索**：同时使用稠密向量（语义相似度）和稀疏向量（关键词匹配），兼顾语义和精确匹配
 - **HyDE 增强**：通过假设性回答编码提升查询召回质量
 - **商品名语义确认**：查询时先进行商品名识别和向量库确认，避免跨商品混淆
 - **动态 Top-K**：Rerank 后基于分数断层智能截断，避免固定截断导致信息丢失或引入噪声
 - **多源 RRF 融合**：稠密检索 + HyDE + 联网搜索结果的加权倒排融合
 - **流式输出**：支持 SSE 长连接逐 token 流式输出答案，提升交互体验
 - **LangGraph 编排**：将复杂 RAG 流程拆解为有向图节点，便于维护和扩展
 - **完整任务追踪**：每个流程节点都有运行状态记录，前端可实时查看处理进度
 
 ---
 
 ## 11. 注意事项
 
 - 所有 LLM 调用通过 OpenAI 兼容 API（阿里云百炼 DashScope），需有效 API Key
 - BGE-M3 和 BGE-Reranker 模型建议使用 GPU 运行以提升速度；CPU 运行需在 `.env` 中设置 `BGE_DEVICE=cpu`
 - MinerU API 为在线服务，PDF 解析消耗网络请求和轮询时间
 - 生产环境部署建议修改 FastAPI CORS 的 `allow_origins=["*"]` 为具体域名
