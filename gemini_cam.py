## source myenv/bin/activate

import cv2
import google.generativeai as genai
from PIL import Image
from google.generativeai.types import content_types
import streamlit as st


API_KEY = 'USE YOUR OWN API KEY'
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro-001")

def get_gemini_pro_vision_response(model, prompt, frames, generation_config={}, stream: bool = True):
    content = []
    content.append(prompt)
    for frame in frames:
        encoded_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        encoded_image = content_types.pil_to_blob(encoded_image)
        content.append(encoded_image)

    generation_config = {"temperature": 0.1, "max_output_tokens": 2048,}
    responses = model.generate_content(
        content, generation_config = generation_config, stream = stream
    )
    responses.resolve()
    return responses.text


def process_video_chunk(frames, user_prompt):
    if not user_prompt :
        prompt = "Given these images, Identify the number of people in the each all these images also tell us if there is somehting suspicions going."
    else :
        prompt = user_prompt
    response = get_gemini_pro_vision_response(model, prompt, frames)
    return response


####################### to fix here 
def feed_video_to_gemini(video_stream, user_prompt):
    chunk_size = 15
    frames = []
    image_counter = 0 

    # Create the stop button once, outside the loop
    stop_button = st.button("Stop", key="stop_button")

    while not stop_button:
        ret, video_chunk = video_stream.read()
        if not ret:
            break

        frames.append(video_chunk)
        if len(frames) == chunk_size:
            with st.spinner("Analyzing video chunk..."):
                try:
                    result = process_video_chunk(frames, user_prompt)
                    st.write("**Analysis Results:**")
                    st.write(result)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            frames = []

        # st.image(video_chunk, channels="BGR", use_column_width=True)
        # Display only every 5th image
        if image_counter % chunk_size == 0:
            st.image(video_chunk, channels="BGR", use_column_width=True)

        image_counter += 1

    video_stream.release()

############################

st.title("Video Analysis with Gemini")

default_prompt = "Given these images, Identify the number of people in the each all these images also tell us if there is somehting suspicions going."
# user_prompt = st.text_input("Enter your prompt:", value=default_prompt, height=100)
user_prompt = st.text_area("Enter your prompt:", value=default_prompt, height=50) 


if st.button("Start Video Analysis"):
    video_stream = cv2.VideoCapture(0)
    feed_video_to_gemini(video_stream, user_prompt)





