import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

model_name = os.getenv("OPENAI_MODEL", "openrouter/free")

print("当前模型：", model_name)
print("Base URL：", os.getenv("OPENAI_BASE_URL"))

response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": "你是一个简洁的中文助手。"},
        {"role": "user", "content": "请用一句话介绍大连海洋大学新生适应智能体。"}
    ],
    temperature=0.2,
)

print("\n模型返回：")
print(response.choices[0].message.content)