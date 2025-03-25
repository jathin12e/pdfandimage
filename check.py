import cv2
import pytesseract
from PIL import Image
import numpy as np
import fitz  # PyMuPDF
import cohere
import gradio as gr
import os
from dotenv import load_dotenv
from pyngrok import ngrok

# Load environment variables from .env file
load_dotenv()

# Retrieve Cohere API key
api_key = os.getenv("COHERE_API_KEY")
if not api_key:
    raise ValueError("❌ Missing Cohere API Key. Please check your environment variables or .env file.")

# Initialize Cohere client
cohere_client = cohere.Client(api_key)

# Set Tesseract Path (Update this path if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Function to extract text from an image
def extract_text_from_image(image):
    """Extract text from an image using Tesseract OCR."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        text = pytesseract.image_to_string(gray)
        return text.strip() if text.strip() else "No readable text found in the image."
    except Exception as e:
        return f"Error processing image: {str(e)}"

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        text = "".join(page.get_text() + "\n" for page in doc)
        return text.strip()
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

# Function to generate responses using Cohere's Chat API
def query_cohere(extracted_text, query):
    """Query Cohere's Chat API with extracted text and a user prompt."""
    prompt = f"""
    You are an AI assistant trained to answer questions based on the provided document or image.
    Here is the extracted text from the input:

    {extracted_text}

    Now, answer this question based on the input:
    {query}
    """

    try:
        response = cohere_client.chat(
            message=prompt,  # The input message for the chat API
            chat_history=[]  # Start a new session for each query
        )

        # Access the generated response text
        return response.text.strip()
    except Exception as e:
        return f"Error in AI processing: {str(e)}"

# Function to process input for both PDF and Image files
def process_input(file, query):
    """Process PDF or image file and query Cohere."""
    if file is None:
        return "Please upload a file."

    file_ext = os.path.splitext(file.name)[1].lower()

    if file_ext == ".pdf":
        # Process PDF
        extracted_text = extract_text_from_pdf(file.name)
        if not extracted_text:
            return "The uploaded PDF does not contain readable text."
    else:
        # Process Image
        try:
            image = Image.open(file).convert("RGB")
            image = np.array(image)
            extracted_text = extract_text_from_image(image)
        except Exception as e:
            return f"Error processing image: {str(e)}"

    if extracted_text == "No readable text found in the image.":
        return "No text found in the uploaded file. Try uploading a clearer file."

    return query_cohere(extracted_text, query)

# Function to preview the uploaded image immediately
def preview_image(file):
    """Preview the uploaded image."""
    if file is None:
        return None  # No file uploaded
    file_ext = os.path.splitext(file.name)[1].lower()
    if file_ext in [".png", ".jpg", ".jpeg"]:
        return Image.open(file)
    return None  # Return None if the file is not an image

# Gradio UI
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            file_input = gr.File(
                label="Upload PDF or Image",
                file_types=[".pdf", ".png", ".jpg", ".jpeg"]
            )
            image_preview = gr.Image(type="pil", label="Image Preview", interactive=False)
            query_input = gr.Textbox(label="Enter your query")
        with gr.Column():
            response_output = gr.Textbox(label="Response")

    # Automatically preview the image when a file is uploaded
    file_input.change(preview_image, inputs=file_input, outputs=image_preview)

    # Button to process the file and query
    submit_button = gr.Button("Submit")
    submit_button.click(
        process_input,
        inputs=[file_input, query_input],
        outputs=response_output
    )

# Run the Gradio app with Ngrok for public URL
if __name__ == "__main__":
    public_url = ngrok.connect(7860)  # Expose port 7860
    print(f"Public URL: {public_url}")
    demo.launch(server_port=7860)
