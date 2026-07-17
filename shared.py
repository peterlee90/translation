from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# 딥러닝 모델 및 인메모리 DB 단일 인스턴스화
EXACT_MATCH_DICT = {}
embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")
qdrant = QdrantClient(":memory:")
