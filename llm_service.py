import os
from openai import AsyncOpenAI

class DeepSeekClient:
    def __init__(self, api_key: str, system_prompt: str = None):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.model = "deepseek-chat" 
        
        self.history = []
        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

    async def get_response(self, user_input: str) -> str:
        self.history.append({"role": "user", "content": user_input})
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                temperature=1.0, 
                stream=False
            )
            reply_text = response.choices[0].message.content
            self.history.append({"role": "assistant", "content": reply_text})
            return reply_text
        except Exception as e:
            print(f"DeepSeek Error: {e}")
            return "Sorry, I lost my connection. Could you repeat that?"