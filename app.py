# app.py
import streamlit as st
import fitz  # PyMuPDF
import re
import math
import os
import tempfile
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==============================================================================
# 1. BACKEND LOGIC (Your existing functions)
# ==============================================================================

def extract_text_with_location(doc):
    """Extracts all text and their locations from a PDF document."""
    extracted_text_with_location = []
    for page_num in range(doc.page_count):
        page = doc[page_num]
        words = page.get_text("words")
        for word in words:
            extracted_text_with_location.append({
                "text": word[4],
                "bbox": (word[0], word[1], word[2], word[3]),
                "page": page_num
            })
    return extracted_text_with_location

def filter_chinese_text(extracted_data):
    """Filters the extracted text to keep only likely Chinese text."""
    extracted_chinese_text_with_location = []
    for item in extracted_data:
        if is_likely_chinese(item["text"]):
            extracted_chinese_text_with_location.append(item)
    return extracted_chinese_text_with_location

def is_likely_chinese(text):
    """Checks if a string likely contains Chinese characters."""
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    return len(chinese_chars) > 0

# Use Streamlit's caching to load the model only once.
@st.cache_resource
def load_model():
    """Loads and caches the translation model and tokenizer."""
    local_model_path = "./offline_model"
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path)
    return tokenizer, model

def translate_chinese_to_english(chinese_text_data, tokenizer, model):
    """Translates extracted Chinese text to English."""
    translated_data = []
    # Create a progress bar for the translation process
    progress_bar = st.progress(0, text="Translating text...")
    total_items = len(chinese_text_data)
    
    for i, item in enumerate(chinese_text_data):
        chinese_text = item["text"]
        try:
            input_ids = tokenizer(chinese_text, return_tensors="pt").input_ids
            translated_ids = model.generate(input_ids, max_length=512)
            english_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True).strip()
        except Exception as e:
            print(f"Error translating '{chinese_text}': {e}")
            english_text = ""
        
        translated_data.append({
            "text": chinese_text,
            "bbox": item["bbox"],
            "page": item["page"],
            "english_translation": english_text
        })
        # Update the progress bar
        progress_bar.progress((i + 1) / total_items, text=f"Translating item {i+1}/{total_items}")
        
    progress_bar.empty() # Clear the progress bar when done
    return translated_data

def get_optimal_fontsize(rect, text, fontname="helv", max_fontsize=12):
    """Calculates the optimal font size to fit text into a rect on a single line."""
    text_len_at_size_1 = fitz.get_text_length(text, fontname=fontname, fontsize=1)
    if text_len_at_size_1 == 0:
        return max_fontsize
    optimal_size = rect.width / text_len_at_size_1
    return min(optimal_size, max_fontsize)

def create_translated_pdf(doc, translated_data, output_filename="translated_output.pdf"):
    """Creates a new PDF with Chinese text replaced by English translations."""
    output_doc = fitz.open()
    for page_num in range(doc.page_count):
        page = doc[page_num]
        output_page = output_doc.new_page(width=page.rect.width, height=page.rect.height)
        output_page.show_pdf_page(page.rect, doc, page_num)

        for item in translated_data:
            if item["page"] == page_num:
                original_bbox = fitz.Rect(item["bbox"])
                english_text = item["english_translation"]
                
                if english_text:
                    output_page.draw_rect(original_bbox, color=(1, 1, 1), fill=(1, 1, 1), overlay=True)
                    best_fsize = get_optimal_fontsize(original_bbox, english_text)
                    output_page.insert_textbox(
                        original_bbox,
                        english_text,
                        fontsize=best_fsize,
                        fontname="helv",
                        color=(0, 0, 0),
                        align=fitz.TEXT_ALIGN_CENTER,
                        overlay=True
                    )
    output_doc.save(output_filename)
    output_doc.close()
    return output_filename

# ==============================================================================
# 2. STREAMLIT USER INTERFACE
# ==============================================================================

# Configure the page
st.set_page_config(layout="wide")
st.title("📄 Vector PDF Translator (Chinese to English)")
st.info("Upload a **vector-based PDF** with Chinese text. The tool will extract the text, translate it to English, and generate a new PDF with the translations overlaid.")

# 1. File Uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # 2. Translate Button
    if st.button("Translate PDF"):
        # Save uploaded file to a temporary location for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name

        # 3. Use st.status to show detailed progress
        with st.status("Processing PDF...", expanded=True) as status:
            try:
                st.write("Loading translation model (this might take a moment on first run)...")
                tokenizer, model = load_model()

                st.write("Step 1/4: Opening PDF document...")
                doc = fitz.open(pdf_path)

                st.write("Step 2/4: Extracting and filtering Chinese text...")
                all_text = extract_text_with_location(doc)
                chinese_text_data = filter_chinese_text(all_text)
                
                if not chinese_text_data:
                    st.warning("No Chinese text was found in this PDF.")
                    status.update(label="Processing complete.", state="complete", expanded=False)
                else:
                    st.write(f"Step 3/4: Translating {len(chinese_text_data)} text fragments...")
                    translated_data = translate_chinese_to_english(chinese_text_data, tokenizer, model)
                    
                    st.write("Step 4/4: Generating final translated PDF...")
                    output_pdf_path = os.path.join(tempfile.gettempdir(), f"translated_{uploaded_file.name}")
                    create_translated_pdf(doc, translated_data, output_pdf_path)
                    
                    doc.close()
                    status.update(label="✅ Translation Complete!", state="complete", expanded=False)
                    
                    # 4. Display success and provide download button
                    st.success("Your PDF has been successfully translated!")
                    
                    with open(output_pdf_path, "rb") as file:
                        st.download_button(
                            label="Download Translated PDF",
                            data=file,
                            file_name=f"translated_{uploaded_file.name}",
                            mime="application/pdf"
                        )

            except Exception as e:
                st.error(f"An error occurred: {e}")
                status.update(label="❌ Error", state="error", expanded=True)