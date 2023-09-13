import openai

openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = "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" #replace with your own api key. 


headers = {
    "HTTP-Referer": "https://localhost"
}

response = openai.ChatCompletion.create(
    #model="meta-llama/llama-2-70b-chat",  
    #model = "anthropic/claude-2",
    model = "openai/gpt-3.5-turbo",	
    messages=[
    {"role": "user", "content": "whats the day today"}
    ],
    headers=headers  # Pass the headers to the API call
)
reply = response.choices[0].message
print(reply.content)