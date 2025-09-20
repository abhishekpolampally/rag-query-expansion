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
client = OpenAI(api_key=openai_key)

reader = PdfReader("data/microsoft-annual-report.pdf")

# Remove trailing whitespaces
pdf_texts = [p.extract_text().strip() for p in reader.pages]

# Filter empty strings
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

chrome_client = chromadb.Client()
chroma_collection = chrome_client.create_collection(name="microsoft-annual-report", embedding_function=embedding_function)

ids = [str(i) for i in range(len(token_split_texts))]
chroma_collection.add(ids=ids, documents=token_split_texts)
chroma_collection.count()

query = "What was the total revenue for the year?"

results = chroma_collection.query(
    query_texts=[query],
    n_results=5,
)
retrieved_documents = results["documents"][0]

def generate_multi_query(query, model="gpt-3.5-turbo", max_tokens=512):
    prompt = """
        You are a knowledgeable expert financial research assistant.
        You users are inquiring about an annual report.
        For the given question, generate a set of 5 diverse and relevant questions that would help in gathering comprehensive information to answer the original question.
        Provide concise, single-topic questions without any additional context or explanations. (without compound sentences)
        Ensure each question is complete and directly related to the original question.
        List each question on a seperate line without numbering or bullet points or another other prefix for list item.
    """

    system_message = {
        "role": "system",
        "content": prompt,
    }

    user_message = {
        "role": "user",
        "content": query,
    }

    response = client.chat.completions.create(
        model=model,
        messages=[system_message, user_message],
        max_tokens=max_tokens,
    )

    content = response.choices[0].message.content.strip().split("\n")
    return content
    
original_query = "What details can you provide about the factors that led to the revenue growth?"
augmented_queries = generate_multi_query(original_query)

joint_query = [original_query] + augmented_queries

results = chroma_collection.query(query_texts=joint_query, n_results=5, include=["documents", "embeddings"])
retrieved_documents = results["documents"]
retrieved_embeddings = results["embeddings"]
result_embeddings = [item for sublist in retrieved_embeddings for item in sublist]

unique_retrieved_docs = set()
for docs in retrieved_documents:
    for doc in docs:
        unique_retrieved_docs.add(doc)

embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

original_query_emedding = embedding_function([original_query])
augmented_query_embeddings = embedding_function(joint_query)

projected_original_query = project_embeddings(
    embeddings=original_query_emedding, umap_transform=umap_transform
)
projected_augmented_queries = project_embeddings(
    embeddings=augmented_query_embeddings, umap_transform=umap_transform
)
projected_result_embeddings = project_embeddings(
    embeddings=result_embeddings, umap_transform=umap_transform
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
    projected_result_embeddings[:, 0],
    projected_result_embeddings[:, 1],
    s=100,
    facecolors="none",
    edgecolors="g",
)
plt.scatter(
    projected_original_query[:, 0],
    projected_original_query[:, 1],
    s=150,
    marker="X",
    color="r",
)
plt.scatter(
    projected_augmented_queries[:, 0],
    projected_augmented_queries[:, 1],
    s=150,
    marker="X",
    color="orange",
)

plt.gca().set_aspect("equal", "datalim")
plt.title(f"${original_query}")
plt.axis("off")
plt.show()