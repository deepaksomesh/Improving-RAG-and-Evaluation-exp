import asyncio
from openai import AsyncOpenAI
import tomli

with open("credentials/credentials.toml", "rb") as f:
    creds = tomli.load(f)

client = AsyncOpenAI(
    api_key=creds['ai']['openai_api_key'],
    base_url=creds['ai']['openai_base_url']
)

async def test():
    try:
        response = await client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        print("Success:", response.choices[0].message.content)
    except Exception as e:
        print("Error:", e)

asyncio.run(test())
