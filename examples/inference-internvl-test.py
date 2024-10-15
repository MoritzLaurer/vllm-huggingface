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

def chat_completions(messages):
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
                    #"url": "https://images.unsplash.com/photo-1529778873920-4da4926a72c2?q=80&w=2853&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
                }
            },
            {
                "type": "text",
                # special prompts for internvl: https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html#grounding-detection-data
                #"text": "Please describe the image in detail."
                #"text": "Please detect and label all objects in the following image and mark their positions."
                "text": "Please provide the bounding box coordinate of the region this sentence describes: <ref>only the yellow part of the bird's chest</ref>"  #"What is in the above image? Explain in detail."
            }
        ]},
    ]
    
    
    start = time()
    response = chat_completions(messages)
    print(f"LLM output: {response}")
    print(f"Time taken: {time() - start:.2f}s")
    
    
    
    bounding_box_detection = True
    if bounding_box_detection == True:
        # parse bounding box coordinates
        import re

        def parse_string(s):
            pattern = r'^(.*?)\s*\[\[([^\]]+)\]\]'
            match = re.match(pattern, s)
            if match:
                text = match.group(1).strip()
                bbox_str = match.group(2)
                bounding_box = [int(num.strip()) for num in bbox_str.split(',')]
                return {"text": text, "bbox": bounding_box}
            else:
                return None
        
        bbox_dic = parse_string(response)

        
        # save image with bounding box
        from PIL import Image, ImageDraw, ImageFont
        import requests
        from io import BytesIO
        
        # https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html#grounding-detection-data
        def scale_coordinates(box, image_width, image_height):
            x1, y1, x2, y2 = box
            scaled_box = [
                int((x1 / 1000) * image_width),
                int((y1 / 1000) * image_height),
                int((x2 / 1000) * image_width),
                int((y2 / 1000) * image_height)
            ]
            return scaled_box

        
        # Download the image
        response = requests.get(messages[1]["content"][0]["image_url"]["url"])
        img = Image.open(BytesIO(response.content))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        image_width, image_height = img.size
        
        # Normalize coordinates
        scaled_bbox = scale_coordinates(bbox_dic["bbox"], image_width, image_height)
        print('Scaled coordinates:', scaled_bbox)

        # Draw on the image
        draw = ImageDraw.Draw(img)

        # Specify the font and increase the font size
        # You can use a common font like Arial. Ensure the font file is accessible.
        font_size = max(20, int(min(image_width, image_height) / 10))  # Adjust font size based on image dimensions
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            # Fallback to the default font if arial.ttf is not found
            font = ImageFont.load_default()

        # Increase the line width for the bounding box
        line_width = max(3, int(min(image_width, image_height) / 150))  # Adjust line width based on image dimensions

        # Draw the bounding box with thicker lines
        draw.rectangle(scaled_bbox, outline='red', width=line_width)

        # Calculate text size using draw.textbbox()
        bbox = draw.textbbox((0, 0), bbox_dic["text"], font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Position the text above the bounding box, adjust if it goes out of image bounds
        text_x = scaled_bbox[0]
        text_y = max(scaled_bbox[1] - text_height - 5, 0)

        # Draw the label text in red and larger font
        draw.text((text_x, text_y), bbox_dic["text"], fill='red', font=font)

        # Save the image
        img.save('output_image.jpg')
        print('Image saved as output_image.jpg')
