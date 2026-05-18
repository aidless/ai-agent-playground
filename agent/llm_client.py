
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import openai
from observability.tracer import log_trace

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError)),
    reraise=True
)
def call_llm_with_retry(client, messages, model, trace_id, step):
    try:
        resp = client.chat.completions.create(model=model, messages=messages)
        log_trace(trace_id, step, "llm_success", {"tokens": resp.usage.total_tokens if resp.usage else 0})
        return resp
    except openai.BadRequestError as e:
        if "context_length" in str(e).lower():
            log_trace(trace_id, step, "context_overflow", {"action": "truncate_fallback"})
            messages = [messages[0]] + messages[-3:]
            return client.chat.completions.create(model=model, messages=messages)
        raise
