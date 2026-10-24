from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

texts = [
    "Apple makes very good computers.",
    "I believe Apple is innovative!",
    "I love apples.",
    "I am a fan of MacBooks.",
    "I enjoy oranges.",
    "1",
    "I like Lenovo Thinkpads.",
    "I think pears taste very good.",
]

# Load the embedding model locally.
# The first run downloads the model; subsequent runs use the local cache.
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# Build the local FAISS vector store.
vector_store = FAISS.from_texts(
    texts,
    embedding=embeddings,
)

# print(vector_store.similarity_search("Apple are my favorite food."))
print(vector_store.similarity_search("Linux is a great operating system."))
