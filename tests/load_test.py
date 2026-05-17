"""Load testing for AI Agent Playground."""

from locust import HttpUser, task, between


class AgentUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def hello_agent(self):
        """Test Hello Agent endpoint."""
        self.client.post(
            "/hello",
            json={"message": "Hello, how are you?"},
            timeout=10
        )

    @task(2)
    def rag_query(self):
        """Test RAG Q&A endpoint."""
        self.client.post(
            "/rag/query",
            json={"question": "What is this document about?"},
            timeout=15
        )

    @task(1)
    def code_review(self):
        """Test Code Review endpoint."""
        self.client.post(
            "/code-review",
            json={
                "code": "def hello(): return 'world'",
                "language": "python"
            },
            timeout=20
        )