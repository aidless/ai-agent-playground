# 刘泽文 — 个人简历

## 基本信息

- **姓名**: 刘泽文
- **学校**: 齐鲁理工学院，软件工程专业，2026 届本科
- **所在地**: 山东
- **GitHub**: github.com/aidless
- **求职方向**: AI Agent 开发工程师 / 大模型应用开发

## 技术栈

- **编程语言**: Python（熟练），Java（Spring Boot）
- **大模型**: DeepSeek API，Anthropic SDK，OpenAI API 兼容接口
- **Agent 框架**: LangChain，MCP 协议（Model Context Protocol），ReAct 循环
- **数据存储**: ChromaDB（向量数据库），MySQL，FAISS
- **后端**: FastAPI，Spring Boot，RESTful API 设计
- **工具链**: Docker，Git，uv（Python 包管理），Streamlit
- **其他**: LoRA 微调，BERT 模型实现，RAG 检索增强生成

## 项目经历

### AI Agent Playground — 多智能体系统（2026.03 - 至今）
- 从 HuggingFace Transformers 源码中提取 `BaseAgent` Pipeline 模式（preprocess → _forward → postprocess），构建了约 200 行的通用 Agent 基类
- 基于基类开发了 5 个 AI Agent：对话助手、代码审查（支持 15+ 语言）、RAG 问答（PDF → ChromaDB → 引用回答）、多智能体协作团队（PM→Dev→QA→DevOps）、简历匹配分析器
- 实现了 MCP 协议（JSON-RPC over stdio）客户端，使 Agent 可以动态发现和调用外部工具，解除了工具硬编码限制
- 集成 Docker 安全沙盒，实现代码执行隔离（网络隔离、内存限制、进程限制、只读文件系统）
- 技术栈：Python，DeepSeek API，Anthropic SDK，ChromaDB，Streamlit，Docker

### Mini BERT — 从零实现 BERT 模型（2026.04）
- 用 Python 从零实现了 BERT 模型的 Transformer 架构，包括多头注意力、位置编码、前馈网络
- 在文本分类任务上完成训练和评估，理解了大模型底层原理
- 技术栈：Python，PyTorch

### LoRA 微调实验（2026.04）
- 使用 LoRA（Low-Rank Adaptation）技术对预训练模型进行参数高效微调
- 对比了全量微调与 LoRA 微调的性能和效率差异
- 技术栈：Python，PyTorch，PEFT

### Java 后端项目（2022）
- 使用 Spring Boot 完成企业级后端项目开发
- 技术栈：Java，Spring Boot，MySQL

## 个人优势

- **动手能力强**: GitHub 上维护 3 个公开项目，仅 ai-agent-playground 就包含 5 个完整 Agent 实现
- **技术写作**: 持续撰写 AI 技术博客，记录学习过程和项目实现细节
- **英语能力**: 可流畅阅读英文技术文档和论文，GitHub README 全部使用英文撰写
- **持续学习**: 2026 年独立完成了从 LLM 基础到 Agent 系统的完整学习路径
