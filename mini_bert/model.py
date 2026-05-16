"""
Mini-BERT —— 从零开始，用 PyTorch 手写一个 Transformer 模型。

这是你理解"AI 大脑内部怎么工作"的关键代码。
每行的张量形状都标注了，方便你在脑子里"运行"这段代码。

先看整体结构（从上往下是数据流）：
  文字 → token ids → [Embedding] → [TransformerBlock × 4] → [Pooler] → [Classifier] → 分类结果

  Embedding:       把"单词编号"变成"有意义的向量"（就像给每个词一个 GPS 坐标）
  TransformerBlock: 理解词与词之间的关系（"苹果"和"吃"关系近，"苹果"和"汽车"关系远）
  Pooler:          从整句话里提取"中心思想"（就像读完一段话后，你能用一句话概括）
  Classifier:      根据中心思想判断类别（正面/负面、体育/科技/商业...）

参考：
  - 原始论文："Attention Is All You Need"（注意力就是一切，2017）
  - HF 源码：E:/transformers-main/src/transformers/models/bert/modeling_bert.py
"""

import math         # 数学函数（sqrt 等）
import torch         # PyTorch 核心库（张量 = 多维数组，就像 NumPy 但能跑在 GPU 上）
import torch.nn as nn           # 神经网络模块（Linear、LayerNorm、Dropout...）
import torch.nn.functional as F  # 函数式 API（softmax、gelu、cross_entropy...）


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  第 1 层：Embedding（嵌入）—— 把"单词编号"变成"有意义的向量"                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 为什么需要 Embedding？
#   计算机不认识"苹果"这个词。它只认识数字。
#   所以我们要把每个词映射到一个数字向量（一串小数），比如：
#     "苹果" → [0.2, -0.5, 0.8, ...]  （768 个小数）
#     "香蕉" → [0.1, -0.6, 0.7, ...]  （和"苹果"很近！因为它们都是水果）
#
# BERT 用了三种 Embedding，加在一起：
#   Word Embedding     = 这个词"是什么"（苹果 vs 香蕉 vs 汽车）
#   Position Embedding = 这个词"在哪儿"（句首 vs 句尾——位置很重要！）
#   Segment Embedding  = 这个词"属于哪句话"（第一句还是第二句？）
#
# 三种加起来 → 一个词就同时有了"身份"+"位置"+"归属"信息。

class BertEmbedding(nn.Module):
    """
    Word + Position + Segment → 一个混合向量。

    就像调鸡尾酒：伏特加(词) + 橙汁(位置) + 冰块(句段) = 一杯鸡尾酒。
    每种原料单独放是一个味道，混在一起才是完整的体验。
    """

    def __init__(self, vocab_size=30522, hidden_size=768,
                 max_position=512, type_vocab_size=2,
                 layer_norm_eps=1e-12, dropout=0.1, pad_token_id=0):
        """
        参数说明：
          vocab_size    = 词汇表大小（有多少种"酒"可选）
          hidden_size   = 隐藏层维度（向量的长度，标准 BERT 用 768）
          max_position  = 最大句子长度（一句话最多多少个词）
          type_vocab_size = 句子类型数（A句还是B句？通常是2）
          dropout       = 随机丢弃率（防过拟合，训练时随机扔掉一些神经元）
          pad_token_id  = 填充符的 ID（0 号位置是 [PAD]）
        """
        super().__init__()

        # ---- 三种 Embedding ----
        # 词汇嵌入：每个单词 → 一个长度为 hidden_size 的向量
        # padding_idx=0 表示 0 号词([PAD])永远映射到零向量（它不携带信息）
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)

        # 位置嵌入：第0个位置、第1个位置... → 各自的向量
        # 为什么需要？"我打你"和"你打我"——词一样，位置不同，意思完全不同！
        self.position_embeddings = nn.Embedding(max_position, hidden_size)

        # 句段嵌入：第一句话还是第二句话？
        # 为什么需要？输入是两句话时（比如问答），模型需要知道哪个词属于哪句
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        # ---- 归一化 + 正则化 ----
        # LayerNorm：把数值范围控制在一个合理区间（不让某些值太大或太小）
        # 就像把所有人的身高统一到"平均身高=0，标准差=1"——方便比较
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # Dropout：训练时随机"关掉"一部分神经元（设为0），防止模型死记硬背
        # 就像老师抽查——不让你背答案，而是理解原理
        self.dropout = nn.Dropout(dropout)

        # ---- 预存位置 ID ----
        # 这是一个"缓存"：提前算好 [0, 1, 2, ..., 511]，不用每次都重新生成
        # register_buffer 表示"这不是模型参数，但需要跟着模型一起走（保存/加载/移动设备）"
        self.register_buffer(
            "position_ids",
            torch.arange(max_position).expand((1, -1)),  # shape: (1, 512)
        )

    def forward(self, input_ids, token_type_ids=None):
        """
        前向传播：输入 token ID → 输出嵌入向量。

        参数:
          input_ids:       (batch_size, seq_len)   比如 (32句话, 每句128个词)
          token_type_ids:  (batch_size, seq_len)   哪个词属于哪句话

        返回:
          embeddings: (batch_size, seq_len, hidden_size)  比如 (32, 128, 768)
        """
        batch_size, seq_length = input_ids.shape

        # 取出位置 ID（只取实际需要的长度）
        # self.position_ids 是 (1, 512)，我们取[:seq_length] → (1, 128)
        position_ids = self.position_ids[:, :seq_length]

        # 如果没给 token_type_ids，默认全填 0（所有词属于第一句）
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                (batch_size, seq_length), dtype=torch.long, device=input_ids.device
            )

        # ---- 三种嵌入，各自查表 ----
        # Embedding 的工作原理：输入一个编号 → 查表 → 返回对应的向量
        # 就像：输入"学生证号 42"→ 查学生名册 → 返回"张三"
        word_emb = self.word_embeddings(input_ids)              # (32, 128, 768)
        pos_emb  = self.position_embeddings(position_ids)       # (1,  128, 768)
        seg_emb  = self.token_type_embeddings(token_type_ids)   # (32, 128, 768)

        # ---- 三者相加：调鸡尾酒！ ----
        # pos_emb 是 (1, 128, 768)，广播到 (32, 128, 768)——每个样本加同样的位置信息
        embeddings = word_emb + pos_emb + seg_emb  # (32, 128, 768)

        # ---- 归一化 + 防过拟合 ----
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings  # (32, 128, 768)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  第 2 层：Multi-Head Self-Attention（多头自注意力）—— Transformer 的核心   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 这是整个 Transformer 最核心的机制，也是最难理解的。
# 我用"图书馆查资料"的类比来解释：
#
# 假设你在图书馆，手里有一句话："苹果很好吃"
# 你想理解"吃"这个词——它和谁关系最密切？
#   - "吃"和"苹果"关系密切（苹果是被吃的）
#   - "吃"和"很"有一点关系（修饰程度）
#   - "吃"和"好"有一些关系（修饰性质）
#
# Self-Attention 做的就是这件事：每个词去"询问"（Query）所有其他词，
# 找到和自己相关的内容，然后从相关词那里"抽取"（Value）信息。
#
# 公式（面试必问！）：Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V
#
#   Q (Query, 查询):  "我想知道什么？" ——当前词发出的"问题"
#   K (Key, 键):      "我是什么？"   ——每个词贴的"标签"
#   V (Value, 值):     "我有什么信息？"——每个词携带的"内容"
#
#   过程：Q 和所有 K 做点积 → 得到"相关性分数"
#         → 除以 √d_k（防止数值爆炸）
#         → softmax（变成 0-1 之间的权重）
#         → 用权重对 V 做加权求和
#
#   为什么叫"多头"？因为一个"头"可能关注语法关系，
#   另一个"头"关注语义关系，多个头并行工作，信息更丰富。
#   就像同时请 12 个专家分析一句话，每人从不同角度分析。

class MultiHeadSelfAttention(nn.Module):
    """多头自注意力——"谁和谁相关"的计算引擎。"""

    def __init__(self, hidden_size=768, num_heads=12, dropout=0.1):
        """
        参数:
          hidden_size: 隐藏层维度（768）
          num_heads:   注意力头数（12个"专家"）
          dropout:     随机丢弃率

        每个头的维度 = hidden_size / num_heads = 768 / 12 = 64
        """
        super().__init__()

        # 确保 hidden_size 能被 num_heads 整除（768 / 12 = 64，很完美）
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads    # 64 —— 每个"专家"处理的维度
        self.scaling = self.head_dim ** -0.5         # 1/√64 = 0.125 —— 缩放因子

        # ---- Q、K、V 的投影层 ----
        # 这三个 Linear 把输入（768维）分别转换成 Q、K、V（都是 768 维）
        # 为什么要投影？原始输入是一个笼统的表示，Q/K/V 各自赋予不同含义
        self.query = nn.Linear(hidden_size, hidden_size)   # 生成"问题"
        self.key   = nn.Linear(hidden_size, hidden_size)   # 生成"标签"
        self.value = nn.Linear(hidden_size, hidden_size)   # 生成"内容"

        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, attention_mask=None):
        """
        输入:
          hidden_states:  (batch, seq_len, hidden_size)  比如 (32, 128, 768)
          attention_mask: (batch, 1, 1, seq_len)  ——填充位置 = 很大的负数，正常位置 = 0

        返回:
          输出: (batch, seq_len, hidden_size)
        """
        bsz, seq_len, _ = hidden_states.shape  # 取 batch_size 和 序列长度

        # ============================================================
        #  步骤 1：投影 —— 生成 Q、K、V
        # ============================================================
        # 把输入通过三个 Linear 层，得到查询、键、值
        Q = self.query(hidden_states)   # (32, 128, 768) ——"我想知道什么？"
        K = self.key(hidden_states)     # (32, 128, 768) ——"我是什么？"
        V = self.value(hidden_states)   # (32, 128, 768) ——"我有什么？"

        # ============================================================
        #  步骤 2：分头 —— 把一个大矩阵拆成多个"专家"视角
        # ============================================================
        # 从 (32, 128, 768) → (32, 128, 12, 64) → (32, 12, 128, 64)
        #                  ↑变成4维      ↑把"头数"维提前，方便批量计算
        Q = Q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # ============================================================
        #  步骤 3：计算注意力分数 —— Q 和 K 的"匹配程度"
        # ============================================================
        # Q: (32, 12, 128, 64)  ×  K^T: (32, 12, 64, 128)  →  (32, 12, 128, 128)
        # 结果矩阵[i][h][a][b] = 第 i 句话，第 h 个头，词 a 对词 b 的"关注程度"
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scaling
        # × scaling (0.125) 是为了防止点积太大，导致 softmax 后梯度消失
        # 为什么点积会太大？64 维向量的点积，期望值是 0，但方差是 64
        # 除以 √64 = 8 后，方差变成 1。乘以 1/8 等价于除以 8

        # ============================================================
        #  步骤 4：掩盖填充位置 —— [PAD] 不应参与注意力
        # ============================================================
        # attention_mask 里填充位置 = -10000，正常位置 = 0
        # 加上 -10000 后，这些位置在 softmax 后会变成 ~0（e^{-10000} ≈ 0）
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # ============================================================
        #  步骤 5：Softmax —— 把分数变成"概率权重"
        # ============================================================
        # 在最后一个维度（被关注词的方向）做 softmax
        # 结果：每行加起来 = 1，每个值在 [0, 1] 之间
        attn_weights = F.softmax(attn_scores, dim=-1)  # (32, 12, 128, 128)
        attn_weights = self.dropout(attn_weights)       # 训练时随机丢一些

        # ============================================================
        #  步骤 6：加权求和 —— 根据"关注谁"来"吸收信息"
        # ============================================================
        # 注意力权重: (32, 12, 128, 128) × V: (32, 12, 128, 64) → (32, 12, 128, 64)
        # 每个词根据它对其他词的关注程度，从其他词那里"吸收"信息
        attn_output = torch.matmul(attn_weights, V)

        # ============================================================
        #  步骤 7：合并头 —— 把 12 个"专家"的意见汇总
        # ============================================================
        # (32, 12, 128, 64) → (32, 128, 12, 64) → (32, 128, 768)
        attn_output = attn_output.transpose(1, 2).contiguous()  # contiguous: 让内存连续，方便 view
        attn_output = attn_output.view(bsz, seq_len, -1)

        return attn_output  # (32, 128, 768) —— 每个词现在"知道了"整句话的上下文


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  第 3 层：Feed-Forward Network（前馈网络）—— 每个词独自"思考"               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Attention 让词之间"交流"，但每个词还需要"独立思考"。
# FFN 就是每个词的"个人学习时间"。
#
# 公式：FFN(x) = GELU(x · W1 + b1) · W2 + b2
#
# 就像：先"放大"（768→3072，给更多思考空间）
#       → 再"压缩"（3072→768，提炼出最重要的信息）
#
# GELU 是什么？ReLU 的"平滑版"。
#   ReLU:  f(x) = x  if x>0, else 0（简单粗暴，但不平滑）
#   GELU:  f(x) = x * Φ(x)（用正态分布平滑过渡，效果更好）

class FeedForward(nn.Module):
    """两层全连接网络：先放大 4 倍，再用 GELU 激活，再缩回来。"""

    def __init__(self, hidden_size=768, intermediate_size=3072, dropout=0.1):
        super().__init__()
        # 第一层：放大（768 → 3072），给模型更多"思考空间"
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        # 第二层：压缩（3072 → 768），提炼最重要的信息
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        """
        输入:  (batch, seq_len, 768)
        输出:  (batch, seq_len, 768)  —— 进和出一样大（方便堆叠）
        """
        h = self.dense1(hidden_states)  # (32, 128, 3072) —— "展开思考"
        h = F.gelu(h)                   # 用 GELU 激活——"添加非线性"
        h = self.dense2(h)              # (32, 128, 768)  —— "提炼精华"
        h = self.dropout(h)             # 防过拟合
        return h


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  第 4 层：Transformer Block —— 一个完整的"思考单元"                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 一个 Transformer Block = Attention（交流） + FFN（思考）
# 每个子层都有"残差连接"（Residual Connection）：
#   输出 = 子层(LayerNorm(输入)) + 输入
#
# 残差连接是什么？就是"抄近道"——把输入直接连到输出。
# 为什么需要？想象 12 层的网络就像一个 12 人的传话游戏。
#   没有残差：第一个人说的，传到第 12 个人已经面目全非
#   有残差：每个人都在原话的基础上只加"修正"，原话始终保留
#
# 数学上：残差让梯度能直接流过，解决了"深层网络训不动"的问题。

class TransformerBlock(nn.Module):
    """
    一个 Transformer 层：
      1. 自注意力（词之间交流）
      2. 前馈网络（每个词独立思考）
      每个子层都有残差连接 + LayerNorm
    """

    def __init__(self, hidden_size=768, num_heads=12,
                 intermediate_size=3072, dropout=0.1):
        super().__init__()
        # 子组件
        self.attention = MultiHeadSelfAttention(hidden_size, num_heads, dropout)  # 交流
        self.ffn = FeedForward(hidden_size, intermediate_size, dropout)           # 思考
        self.ln1 = nn.LayerNorm(hidden_size)  # Attention 前的归一化
        self.ln2 = nn.LayerNorm(hidden_size)  # FFN 前的归一化

    def forward(self, hidden_states, attention_mask=None):
        """
        输入:  hidden_states  (batch, seq_len, hidden_size)
        返回:  hidden_states  (batch, seq_len, hidden_size) —— 形状不变
        """
        # ---- 子层 1：自注意力 + 残差 ----
        # 先归一化 → 做注意力 → 加回原始输入（这是残差！）
        normed = self.ln1(hidden_states)                  # 归一化（让数据稳定）
        attn_out = self.attention(normed, attention_mask)  # 注意力（词之间交流）
        hidden_states = hidden_states + attn_out           # 残差：原话 + 新信息

        # ---- 子层 2：前馈网络 + 残差 ----
        normed = self.ln2(hidden_states)   # 归一化
        ffn_out = self.ffn(normed)         # FFN（独立思考）
        hidden_states = hidden_states + ffn_out  # 残差：原话 + 思考结果

        return hidden_states


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  第 5 层：Encoder —— 把多个 Transformer Block 叠起来                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# BERT 有 12 层（标准版）或 4 层（我们的 mini 版）。
# 每一层都做同样的事情（Attention + FFN），但学到的模式不同：
#   底层（1-3 层）：学语法（主谓宾结构）
#   中层（4-8 层）：学语义（同义词、反义词）
#   高层（9-12层）：学推理（因果关系、逻辑关系）

class BertEncoder(nn.Module):
    """把 N 个 TransformerBlock 串起来，像串糖葫芦一样。"""

    def __init__(self, num_layers=12, hidden_size=768, num_heads=12,
                 intermediate_size=3072, dropout=0.1):
        super().__init__()
        # nn.ModuleList 是特殊的列表——PyTorch 知道里面是模型参数，会正确追踪
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, intermediate_size, dropout)
            for _ in range(num_layers)  # 创建 num_layers 个相同的块
        ])

    def forward(self, hidden_states, attention_mask=None):
        """逐层传递：第 1 层的输出 → 第 2 层的输入 → ... → 最终输出"""
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  第 6 层：Pooler —— 把整句话"压缩"成一个向量                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# BERT 有一个特殊的 [CLS] token，放在句首。
# 训练时，BERT 学会了把整句话的"中心思想"存在 [CLS] 位置。
# Pooler 的工作：取出 [CLS] → 过一个线性层 → tanh 激活 → 输出一个向量。
#
# 这个向量就是"这句话说了什么"的数学表示。
# 之后接一个 Classifier（分类器）就能判断"这是正面评论还是负面评论"。

class BertPooler(nn.Module):
    """取 [CLS] token → 压缩 → 输出一句的中心思想。"""

    def __init__(self, hidden_size=768):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)  # 再加工一下

    def forward(self, hidden_states):
        """
        输入:  (batch, seq_len, hidden_size)
        输出:  (batch, hidden_size) —— 每句话变成一个向量
        """
        # [:, 0] = 所有 batch，第 0 个 token = [CLS]
        first_token = hidden_states[:, 0]  # (32, 768)
        pooled = self.dense(first_token)   # 线性变换
        pooled = torch.tanh(pooled)        # tanh 压缩到 [-1, 1]
        return pooled


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  第 7 层：完整模型 —— 把所有东西拼起来                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class MiniBertForClassification(nn.Module):
    """
    从零开始的 BERT 分类模型。

    数据流：
      token ids → [Embedding] → [Encoder] → [Pooler] → [Classifier] → 分类结果

    大小选择：
      tiny:  hidden=128, layers=2  (约 1M 参数，秒级训练，纯测试用)
      mini:  hidden=256, layers=4  (约 3M 参数，CPU 可训练，推荐)
      base:  hidden=768, layers=12 (约 110M 参数，需 GPU)
    """

    def __init__(self, num_labels=4, hidden_size=256, num_layers=4,
                 num_heads=4, intermediate_size=1024, vocab_size=30522,
                 max_position=512, dropout=0.1):
        """
        参数:
          num_labels:   分类类别数（比如 4 = {体育, 科技, 商业, 世界}）
          hidden_size:  隐藏维度（256 = mini 版）
          num_layers:   层数（4 = mini 版）
          num_heads:    注意力头数
          intermediate_size: FFN 中间层大小（通常是 hidden_size × 4）
          vocab_size:   词汇表大小（BERT 标准是 30522）
          max_position: 最大序列长度
          dropout:      防过拟合比例
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels

        # ---- 组装组件（乐高式拼接！） ----
        self.embeddings = BertEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position=max_position,
            dropout=dropout,
        )
        self.encoder = BertEncoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            dropout=dropout,
        )
        self.pooler = BertPooler(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)  # 最后一层：→ 分类

        # ---- 初始化权重 ----
        # 为什么需要初始化？如果权重全是 0，梯度永远是 0，模型学不动。
        # 好的初始化让模型一开始就站在"起跑线附近"，训练更快。
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """给每种层设置合理的初始值。"""
        if isinstance(module, nn.Linear):
            # 线性层：用正态分布（均值 0，标准差 0.02）
            # 0.02 是 BERT 论文里的经验值，不能太大也不能太小
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)  # 偏置从 0 开始
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)    # LayerNorm 的缩放从 1 开始
            nn.init.zeros_(module.bias)     # LayerNorm 的偏移从 0 开始

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        前向传播：输入 token IDs → 输出分类结果。

        参数:
          input_ids:      (batch, seq_len)   token 编号
          attention_mask: (batch, seq_len)   1=真词, 0=填充（需要忽略）
          labels:         (batch,)           真实类别（训练时用，推理时不传）

        返回:
          dict: {"loss": 损失值(训练时有), "logits": 分类得分, "pooled": 句子向量}
        """
        batch_size, seq_len = input_ids.shape

        # ---- 创建 attention mask ----
        # 把 (batch, seq_len) 的 0/1 mask →
        #     (batch, 1, 1, seq_len) 的 {0, -10000} mask
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=input_ids.device)

        # 扩维：(32, 128) → (32, 1, 1, 128)
        extended_mask = attention_mask[:, None, None, :]
        # 转换：1→0（可以关注），0→-10000（不能关注，softmax 后会变成 ~0）
        extended_mask = (1.0 - extended_mask) * -10000.0

        # ---- 四步 Pipeline（和整个项目风格一致！） ----
        # 1. 嵌入：词 → 向量
        emb = self.embeddings(input_ids, token_type_ids)        # (32, 128, 256)

        # 2. 编码：通过 4 层 Transformer，理解上下文
        encoded = self.encoder(emb, extended_mask)               # (32, 128, 256)

        # 3. 池化：取 [CLS] token，压缩成"句子向量"
        pooled = self.pooler(encoded)                            # (32, 256)

        # 4. 分类：句子向量 → 各类别的得分
        logits = self.classifier(pooled)                         # (32, 4)

        # ---- 计算损失（训练时） ----
        # 交叉熵 = "预测"和"真实答案"之间的差距。
        # 就像：老师说答案是 B，你回答是 A——差距大，惩罚重
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits, "pooled": pooled}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  快速测试：如果你是直接运行这个文件，会执行下面的代码验证模型能跑通           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print("Testing Mini-BERT forward pass...\n")

    # 创建一个超小的 BERT 做测试（128 维度，2 层，2 头）
    model = MiniBertForClassification(
        num_labels=4,
        hidden_size=128,
        num_layers=2,
        num_heads=2,
        intermediate_size=512,
    )

    # 假数据：4 句话，每句 16 个词，分类标签随机
    input_ids = torch.randint(0, 30522, (4, 16))  # 随机 token ID
    attention_mask = torch.ones(4, 16)              # 全部有效（没有填充）
    labels = torch.randint(0, 4, (4,))              # 随机标签

    # 前向传播！
    output = model(input_ids, attention_mask, labels=labels)

    # 验证每个阶段的形状是否正确
    print(f"  Input:        {input_ids.shape}")                     # (4, 16)
    print(f"  Embedding:    (4, 16, {model.hidden_size})")
    print(f"  Encoded:      (4, 16, {model.hidden_size})")
    print(f"  Pooled:       (4, {model.hidden_size})")
    print(f"  Logits:       {output['logits'].shape}")               # (4, 4)
    print(f"  Loss:         {output['loss']:.4f}")

    # 统计参数量
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters:   {n_params:,}")

    print("\n[OK] Forward pass successful!")
