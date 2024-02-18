import streamlit as st
from PIL import Image
import asyncio
import openai
import time
import numpy as np
import json
import base64


# class Message(Model):
#     message: str
from openai import OpenAI
client = OpenAI(api_key="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")
# SEED_PHRASE = "agent1q2pgplfp2su2q320kz2x2w46fyua25wys9ast6gk7rc7crwan8h765ufx9e"
# AGENT_MAILBOX_KEY = "agent1q2pgplfp2su2q320kz2x2w46fyua25wys9ast6gk7rc7crwan8h765ufx9e"
# agent = Agent(
#     name="LightingRec",
#     seed=SEED_PHRASE,
#     mailbox=f"{AGENT_MAILBOX_KEY}@https://agentverse.ai",
# )

async def ask_gpt(prompt):
    response = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-1106:personal::8tLPRAmo",
        messages=[
            {"role": "system", "content": "You are a lighting assistant"},
            {"role": "user", "content": prompt}],
            temperature=0.9
    )
    response_content = response.choices[0].message.content
    return response_content

st.markdown("""
    <h1 style='color: black;'>
        <span style='color: #CCAC00;'>Lumi</span>nate
    </h1>
    """, unsafe_allow_html=True)
st.write("""
    *Customizing a deeper aspect to your images, the lighting* _direction_.
""")
image=None

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption='Image uploaded successfully!', use_column_width=True)
    

def model(image, instruction):
    pass



# Chatbot Section
user_input = st.text_input("How would you like the lighting changed?", "")
if user_input:
    with st.spinner('Processing...'):
        response_text = asyncio.run(ask_gpt(user_input))
    st.text_area("Response", value=response_text, height=100, max_chars=None)
    if image is not None:
        # result = asyncio.run(model(image, response_text))
        # st.image(result, caption='Your new image', use_column_width=True)
        st.write("yes image") #temporary, please delete
    else:
        st.write("No image") 





# async def process_image(image_path):
#     with Image.open(image_path) as img:
#         # Convert the image to grayscale to simplify brightness analysis
#         gray_img = img.convert('L')
#         # Convert to numpy array for analysis
#         img_array = np.array(gray_img)
#     return img_array

# def save_uploaded_file(uploaded_file):
#     try:
#         with open(f"./tempDir/{uploaded_file.name}", "wb") as f:
#             f.write(uploaded_file.getbuffer())
#         return f"./tempDir/{uploaded_file.name}"
#     except Exception as e:
#         st.error(f"Failed to save the uploaded image: {e}")
#         return None

# async def handle_image_upload(uploaded_file):
#     saved_path = save_uploaded_file(uploaded_file)
#     if saved_path:
#         # Process the image asynchronously
#         return await process_image(saved_path)
#     return "Failed to process the image."


# def serialize_image_array(img_array):
#     data_b64 = base64.b64encode(img_array.tobytes()).decode('utf-8')
#     serialized_data = json.dumps({
#         "data": data_b64,
#         "shape": img_array.shape,
#         "dtype": str(img_array.dtype)
#     })
#     return serialized_data

# async def send_image_to_agent(ctx: Context, image_path):
#     # Process the image to get a numpy array
#     img_array = await process_image(image_path)

#     # Serialize img_array for transmission. Example: JSON, base64, etc.
#     # Here, we'll pretend to convert it to a JSON string (you'll need to implement this based on your data)
#     serialized_img = serialize_image_array(img_array)

#     # Send the processed image to the agent
#     ctx.logger.info("Sending processed image data to the agent")
#     await ctx.send(SEED_PHRASE, Message(data=serialized_img))

