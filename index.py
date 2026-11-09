import os
from sentence_transformers import SentenceTransformer
import chromadb

print("Загрузка модели для эмбеддингов...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Подключение к базе...")
client = chromadb.PersistentClient(path="./chroma_db")
# Удаляем старую базу, если она есть
if "docs" in [c.name for c in client.list_collections()]:
    client.delete_collection("docs")
col = client.get_or_create_collection("docs")

chunks = []
metadatas = []

# Читаем все txt файлы из папки docs
for filename in os.listdir("docs"):
    if filename.endswith(".txt"):
        with open(f"docs/{filename}", "r", encoding="utf-8") as f:
            text = f.read()
            # Разбиваем текст на куски по 500 символов
            for i in range(0, len(text), 500):
                chunks.append(text[i:i+500])
                metadatas.append({"source": filename})

print(f"Найдено {len(chunks)} фрагментов. Индексация...")

col.add(
    documents=chunks,
    metadatas=metadatas,
    ids=[str(i) for i in range(len(chunks))]
)
print("Успешно проиндексировано!")