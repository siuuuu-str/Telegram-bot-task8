import subprocess
import asyncio
import logging
import requests
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.client.session.aiohttp import AiohttpSession
import chromadb
from sentence_transformers import SentenceTransformer

TOKEN = "9259216256:AAEqlZ0fvK_gzX5nzUnPg_xVjK6Fahcm-TU"

logging.basicConfig(level=logging.INFO)
session = AiohttpSession()
bot = Bot(token=TOKEN, session=session)
dp = Dispatcher()

print("Загрузка модели...")
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
col = client.get_collection("docs")

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Привет! Я RAG-бот. Я отвечаю на вопросы ТОЛЬКО на основе документов в папке docs. Спроси меня о Python.")

@dp.message()
async def rag_handler(message: types.Message):
    query = message.text
    await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")

    # 1. Ищем 3 самых похожих куска в базе
    results = col.query(query_texts=[query], n_results=3)
    context = "\n".join(results["documents"][0])
    source = results["metadatas"][0][0]["source"]

    # 2. Формируем промпт
    prompt = f"""Ответь на вопрос пользователя, используя ТОЛЬКО следующий текст:
{context}

Вопрос: {query}
Если в тексте нет ответа, напиши 'Я не знаю'.
"""

    try:
        # 3. Отправляем в Ollama
        response = requests.post("http://localhost:11434/api/chat", json={
            "model": "llama3.2",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        })
        answer = response.json()["message"]["content"]
        
        # 4. Отвечаем пользователю и добавляем источник
        await message.answer(f"{answer}\n\n📚 Источник: {source}")
    except Exception as e:
        await message.answer(f"Ошибка: {e}")

@dp.message(Command("reload"))
async def cmd_reload(message: types.Message):
    await message.answer("Начинаю переиндексацию документов...")
    subprocess.run(["python", "index.py"])
    await message.answer("Документы успешно переиндексированы!")

async def main():
    print("RAG-бот запущен!")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())