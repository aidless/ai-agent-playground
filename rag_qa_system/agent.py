"""
RAGAgent —— 基于文档的问答系统。

RAG = Retrieval-Augmented Generation（检索增强生成）

普通 AI 问答：你问 → AI 凭记忆回答（可能记错、可能过时）
RAG 问答：    你问 → 先查文档库 → 把相关段落找出来 → AI 根据原文回答（准确、可溯源）

就像：
  普通学生：老师问"光合作用是什么？"→ 学生凭记忆回答
  RAG 学生：  老师问"光合作用是什么？"→ 学生翻课本找到相关段落 → 照着课本回答

这个 Agent 的工作流：
  1. 用户上传 PDF/TXT 文档
  2. 文档被切成小段 → 转成向量 → 存到 ChromaDB
  3. 用户提问 → 在向量库中找最相关的段落 → 和问题一起发给 AI → AI 给出带引用的回答
"""

from typing import Any

from ai_agent_playground.base import BaseAgent

from .config import RAGConfig
from .ingest import DocumentIngester  # 文档"吃进去"（PDF → 向量）
from .query import RAGQuerier        # 检索 + 生成回答


class RAGAgent(BaseAgent):
    """
    能"读文档"的 AI——上传文件后，可以向它提问文档内容。

    两步使用：
      1. agent.ingest("docs/")  → 加载文档到向量库
      2. agent.ask("什么是 RAG？") → 提问，得到带引用的回答

    Pipeline:
      preprocess:  接收问题
      _forward:    去 ChromaDB 查相关段落 → 和问题一起发给 AI
      postprocess: 格式化回答（加来源标注）
    """

    config_class = RAGConfig

    def __init__(self, config: RAGConfig | None = None):
        super().__init__(config)
        # 两个"手下"：一个管"吃进去"，一个管"查出来"
        self.ingester = DocumentIngester(self.config)  # 文档 → 向量
        self.querier = RAGQuerier(self.config, self.llm)  # 搜索 + AI 回答

    # ============================================================
    #  三步 Pipeline
    # ============================================================

    def preprocess(self, inputs: str, **kwargs) -> dict[str, Any]:
        """接收用户问题。这步很简单——只是把问题放进字典。"""
        return {"question": inputs}

    def _forward(self, model_inputs: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        核心步骤：检索 + 生成。

        1. 去 ChromaDB 找最相关的文档段落（向量相似度搜索）
        2. 把找到的段落和用户问题一起发给 AI
        3. AI 根据原文回答问题（减少幻觉，提高准确性）

        返回: {"question": ..., "answer": ..., "sources": [...], "chunks_retrieved": N}
        """
        result = self.querier.ask(model_inputs["question"])
        return {
            "question": result.question,
            "answer": result.answer,
            "sources": result.sources,          # 用到了哪些源文件
            "chunks_retrieved": result.chunks_retrieved,  # 检索到多少个相关段落
        }

    def postprocess(self, model_outputs: dict[str, Any], **kwargs) -> str:
        """
        格式化输出：问题 + 回答 + 来源列表。

        输出示例：
          Q: 什么是 RAG？
          RAG 是检索增强生成...（详细回答）

          Sources:
            - docs/ai_basics.txt
            (5 chunks retrieved)
        """
        lines = [
            f"Q: {model_outputs['question']}",
            "",
            model_outputs["answer"],
            "",
        ]
        if model_outputs["sources"]:
            lines.append("Sources:")
            for s in model_outputs["sources"]:
                lines.append(f"  - {s}")
            lines.append(f"  ({model_outputs['chunks_retrieved']} chunks retrieved)")
        return "\n".join(lines)

    # ============================================================
    #  高级方法
    # ============================================================

    def ingest(self, path: str):
        """
        加载文档：把 PDF/TXT 文件"喂"给系统。

        工作过程：
          1. 找到目录下所有 PDF/TXT 文件
          2. 读取内容
          3. 切成小段（每段 ~800 字）
          4. 用 ChromaDB 的嵌入模型转成向量
          5. 存到向量数据库

        之后就可以用 ask() 提问了。
        """
        return self.ingester.ingest(path)

    def ask(self, question: str) -> str:
        """问一个问题，从文档中找到答案。"""
        return self.run(question)

    def chat(self):
        """
        交互式问答模式——像聊天一样提问。

        特殊命令：
          sources <query>  → 查看检索结果（调试用，看搜到了什么段落）
          quit             → 退出
        """
        stats = self.ingester.stats()
        if not stats:
            print("No documents ingested. Use 'ingest <path>' first.\n")
            return

        print("=" * 60)
        print(f"  RAG Q&A — {stats['chunks']} chunks in '{stats['name']}'")
        print("  Type 'sources <query>' to debug, 'quit' to exit")
        print("=" * 60)
        print()

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() == "quit":
                print("Goodbye!")
                break

            # 调试命令：查看检索到了哪些段落
            if user_input.lower().startswith("sources "):
                q = user_input[8:]  # 去掉 "sources " 前缀
                chunks = self.querier.search(q)
                for c in chunks:
                    print(f"  [Chunk {c['chunk_index']}] {c['source']} "
                          f"(distance: {c['distance']:.3f})")
                    print(f"    {c['text']}\n")
                continue

            print(self.run(user_input))
            print()
