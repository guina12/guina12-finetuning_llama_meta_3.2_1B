import os
from openai import OpenAI
from evaluation.openai_api.openai_api_key import get_openai_key

#----------------------------------------------------------------------------------------------------------------#

class OpenAIApiClient:
    def __init__(self):
        get_openai_key()
        self.client = OpenAI(
            api_key = os.getenv("OPENAI_API_KEY")
        )

    def generate(self, prompt):
        r = self.client.chat.completions.create(
            model = "gpt-4.1-mini",
            messages = [{"role": "user", "content": prompt}]
        )
        return r.choices[0].message.content
    

#---------------------------------------------------------------------------------------------------------------------#