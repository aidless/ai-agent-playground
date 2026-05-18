"""Agent 消息总线 — 多 Agent 通信基础设施

三种通信模式:
    - direct:  agent → 指定 agent（点对点）
    - broadcast: agent → 所有 agent（广播）
    - delegate: master → worker 委派任务并等待结果

每条消息带 type、sender、receiver、payload 和时间戳。
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MsgType(str, Enum):
    DIRECT = "direct"         # 点对点
    BROADCAST = "broadcast"   # 广播
    DELEGATE = "delegate"     # 委派（需要响应）
    RESULT = "result"         # 委派结果
    ANNOUNCE = "announce"     # 宣告（注册、能力声明）


@dataclass
class Message:
    """总线消息"""
    msg_type: MsgType
    sender: str
    receiver: str = "*"       # "*" = 广播
    payload: Any = None
    timestamp: float = field(default_factory=time.time)
    reply_to: Optional[str] = None  # 回复消息 ID（用于请求-响应配对）


@dataclass
class Envelope:
    """带 ID 的消息信封"""
    id: str
    message: Message
    response: Optional[Any] = None  # 委派的返回结果填这里


class MessageBus:
    """异步消息总线

    每个 Agent 注册自己的信箱（asyncio.Queue），
    总线负责路由消息到正确的信箱。
    """

    def __init__(self):
        self._mailboxes: dict[str, asyncio.Queue] = {}
        self._history: list[Envelope] = []
        self._pending_delegations: dict[str, asyncio.Future] = {}
        self._msg_counter = 0

    def register(self, agent_name: str):
        """注册一个 Agent 到总线"""
        if agent_name not in self._mailboxes:
            self._mailboxes[agent_name] = asyncio.Queue()
            logger.info("[Bus] %s registered", agent_name)

    def unregister(self, agent_name: str):
        """从总线注销"""
        self._mailboxes.pop(agent_name, None)
        logger.info("[Bus] %s unregistered", agent_name)

    async def send(self, sender: str, receiver: str, payload: Any, msg_type: MsgType = MsgType.DIRECT) -> str:
        """发送消息到指定 Agent，返回消息 ID"""
        self._msg_counter += 1
        msg_id = f"msg-{self._msg_counter}"
        msg = Message(msg_type=msg_type, sender=sender, receiver=receiver, payload=payload)
        env = Envelope(id=msg_id, message=msg)
        self._history.append(env)

        if receiver in self._mailboxes:
            await self._mailboxes[receiver].put(env)
        else:
            logger.warning("[Bus] unknown receiver: %s", receiver)

        return msg_id

    async def broadcast(self, sender: str, payload: Any, exclude: list[str] | None = None) -> list[str]:
        """广播消息到所有 Agent（排除指定列表），返回消息 ID 列表"""
        exclude = exclude or []
        ids = []
        for name in self._mailboxes:
            if name not in exclude:
                mid = await self.send(sender, name, payload, MsgType.BROADCAST)
                ids.append(mid)
        return ids

    async def delegate(self, sender: str, receiver: str, payload: Any, timeout: float = 120.0) -> Any:
        """委派任务并等待结果（同步语义，异步实现）"""
        self._msg_counter += 1
        mid = f"delegate-{self._msg_counter}"
        msg = Message(msg_type=MsgType.DELEGATE, sender=sender, receiver=receiver, payload=payload, reply_to=mid)
        env = Envelope(id=mid, message=msg)
        self._history.append(env)

        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending_delegations[mid] = future

        if receiver not in self._mailboxes:
            raise ValueError(f"Receiver not registered: {receiver}")

        await self._mailboxes[receiver].put(env)

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending_delegations.pop(mid, None)
            raise TimeoutError(f"Delegation to {receiver} timed out after {timeout}s")

    def respond(self, delegation_id: str, result: Any):
        """Worker 回复委派结果"""
        future = self._pending_delegations.pop(delegation_id, None)
        if future and not future.done():
            future.set_result(result)
        else:
            logger.warning("[Bus] no pending delegation for: %s", delegation_id)

    async def receive(self, agent_name: str, timeout: float | None = None) -> Envelope | None:
        """Agent 从自己的信箱收取消息"""
        if agent_name not in self._mailboxes:
            return None
        try:
            if timeout:
                return await asyncio.wait_for(self._mailboxes[agent_name].get(), timeout=timeout)
            return await self._mailboxes[agent_name].get()
        except asyncio.TimeoutError:
            return None

    def list_agents(self) -> list[str]:
        """列出所有注册的 Agent"""
        return list(self._mailboxes.keys())

    @property
    def message_count(self) -> int:
        return len(self._history)
