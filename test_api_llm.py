import requests

resp = requests.post(
    "http://127.0.0.1:8000/ask_llm",
    json={
        "question": "校园卡怎么绑定",
        "campus": "黑石礁",
        "stage": "报到当天"
    }
)

print(resp.status_code)
print(resp.json())