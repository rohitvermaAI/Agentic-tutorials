from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI()
with open("Prompt_Engg/prompt.txt", encoding="utf-8") as file:
    prompt = file.read().strip()
response = client.responses.create(
    model="gpt-5",
    input=prompt
)

print(response.output_text)