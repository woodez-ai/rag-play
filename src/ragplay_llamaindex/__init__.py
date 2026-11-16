"""Phase 9: the same assistant rebuilt with LlamaIndex, to compare with the hand-built one.

Mapping (hand-built -> LlamaIndex):
  text2sql.ask          -> NLSQLTableQueryEngine over SQLDatabase (read-only connection)
  store.find / search   -> VectorStoreIndex on PGVectorStore + MetadataFilters
  agent.ask (tool loop) -> FunctionAgent (workflow) with FunctionTools
  anomalies.*           -> the same deterministic functions, wrapped as FunctionTools
Not ported on purpose: the calculation router, filter relaxation and argument checks. This is
the framework's idiomatic setup, measured on the same eval.

PRIVACY: LlamaIndex defaults to OpenAI for the LLM and embeddings. Both are set to local Ollama
on import (`Settings`), before anything else can fall back to them.
"""

from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

LLM_MODEL = "qwen3:8b"
EMBED_MODEL = "mxbai-embed-large"
MXBAI_QUERY = "Represent this sentence for searching relevant passages: "

Settings.llm = Ollama(model=LLM_MODEL, temperature=0, thinking=False, context_window=8192,
                      request_timeout=300)
Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL, query_instruction=MXBAI_QUERY)
