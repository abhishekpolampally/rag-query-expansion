from helper_utils import project_embeddings, word_wrap
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
import os
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import umap

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

reader = PdfReader("data/microsoft-annual-report.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]

pdf_texts = [text for text in pdf_texts if text]

character_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0, separators=["\n\n", "\n", ". ", " ", ""]
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
)

token_split_texts = []
for text in character_split_texts:
    token_split_texts.extend(token_splitter.split_text(text))

embedding_function = SentenceTransformerEmbeddingFunction()

client = chromadb.Client()
chroma_collection = client.create_collection(name="microsoft-annual-report", embedding_function=embedding_function)

ids = [str(i) for i in range(len(token_split_texts))]
chroma_collection.add(ids=ids, documents=token_split_texts)

def augment_query_generated(query, model="gpt-3.5-turbo", max_tokens=512):
    client = OpenAI(api_key=openai_key)

    system_message = {
        "role": "system",
        "content": "You are a helpful expert financial research assistant. Provide an example answer to the given question, that might be found in an annual report.",
    }

    user_message = {
        "role": "user",
        "content": query,
    }

    response = client.chat.completions.create(
        model=model,
        messages=[system_message, user_message],
        max_tokens=max_tokens,
        temperature=0,
    )

    return response.choices[0].message.content

original_query = "What was the total profit for this year, and how does it compare to last year?"
hypothetical_answer = augment_query_generated(original_query)
joint_query = f"{original_query} {hypothetical_answer}"

results = chroma_collection.query(query_texts=[joint_query], n_results=5, include=["documents", "embeddings"])
retrieved_docs = results["documents"][0]

embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
projected_dataset_embeddings = project_embeddings(embeddings=embeddings, umap_transform=umap_transform)

retrieved_embeddings = results["embeddings"][0]
original_query_emedding = embedding_function([original_query])
augmented_query_embedding = embedding_function([joint_query])

projected_original_query_embedding = project_embeddings(
    embeddings=original_query_emedding, umap_transform=umap_transform
)
projected_augmented_query_embedding = project_embeddings(
    embeddings=augmented_query_embedding, umap_transform=umap_transform
)
projected_retrieved_embeddings = project_embeddings(
    embeddings=retrieved_embeddings, umap_transform=umap_transform
)

import matplotlib.pyplot as plt

plt.figure()

plt.scatter(
    projected_dataset_embeddings[:, 0],
    projected_dataset_embeddings[:, 1],
    s=10,
    color="lightgray",
)
plt.scatter(
    projected_retrieved_embeddings[:, 0],
    projected_retrieved_embeddings[:, 1],
    s=100,
    facecolors="none",
    edgecolors="g",
)
plt.scatter(
    projected_original_query_embedding[:, 0],
    projected_original_query_embedding[:, 1],
    s=150,
    marker="X",
    color="r",
)
plt.scatter(
    projected_augmented_query_embedding[:, 0],
    projected_augmented_query_embedding[:, 1],
    s=150,
    marker="X",
    color="orange",
)

plt.gca().set_aspect("equal", "datalim")
plt.title(f"${original_query}")
plt.axis("off")
plt.show()