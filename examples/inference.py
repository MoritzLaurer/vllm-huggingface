from openai import OpenAI
import os
from dotenv import load_dotenv
from time import time

load_dotenv() 
ENDPOINT_URL = os.getenv("HF_ENDPOINT_URL") + "/v1/" # if endpoint object is not available check the UI 
API_KEY = os.getenv("HF_TOKEN")

# initialize the client but point it to TGI
client = OpenAI(base_url=ENDPOINT_URL, api_key=API_KEY)

generation_parameters = {
    "temperature": 0.2,
    "max_tokens": 128,
    "top_p": 0.7,
    "stream": False,
}

def predict(messages):
    return client.chat.completions.create(
        model="/repository", # needs to be /repository since there are the model artifacts stored
        messages=messages,
        **generation_parameters
    ).choices[0].message.content


if __name__ == "__main__":    

    messages = [
        {"role": "system", 
         "content": "You are a helpful assistant",
        },
        {"role": "user", "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://unsplash.com/photos/ZVw3HmHRhv0/download?ixid=M3wxMjA3fDB8MXxhbGx8NHx8fHx8fDJ8fDE3MjQ1NjAzNjl8&force=true&w=1920"
                }
            },
            {
                "type": "text",
                "text": "What type of animal is in the image above? What color does it have?"
                #"text": "\nPlease provide the bounding box coordinate of the region this sentence describes: <ref>the bird</ref>"  #"What is in the above image? Explain in detail."
            }
        ]},
    ]

    start = time()
    response = predict(messages)
    print(f"LLM output: {response}")
    print(f"Time taken: {time() - start:.2f}s")

