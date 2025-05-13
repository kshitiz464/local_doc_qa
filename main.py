import streamlit as st
import pdfplumber
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from llama_cpp import Llama
from transformers import pipeline
import os
import uuid

# --- Initialize models using st.cache_resource to run only once ---
@st.cache_resource
def load_models():
    st.write("Initializing SentenceTransformer model...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    st.write("Initializing Llama model...")
    llm = Llama(model_path="models/gpt4all-falcon-q4_0.gguf", n_ctx=1024, n_threads=2)
    st.write("Initializing Paraphraser pipeline...")
    paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")
    st.write("All models initialized.")
    return embed_model, llm, paraphraser

# Call the cached function to get the models
embed_model, llm, paraphraser = load_models()

# Text extraction
def extract_text(path):
    with st.spinner("Extracting text from document..."): # Added spinner
        if path.endswith(".pdf"):
            with pdfplumber.open(path) as pdf:
                text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        elif path.endswith(".docx"):
            doc = Document(path)
            text = "\n".join(p.text for p in doc.paragraphs)
        else:
            text = "" # Should not happen with type=["pdf", "docx"]
    return text

# Split into chunks
def chunk_text(text, chunk_size=768):
    with st.spinner("Chunking document..."): # Added spinner
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# Build FAISS index
def build_index(chunks):
    with st.spinner("Building FAISS index (this might take a while for large documents)..."): # Added spinner
        embs = embed_model.encode(chunks, convert_to_numpy=True)
        index = faiss.IndexFlatL2(embs.shape[1])
        index.add(embs)
    return index

# Paraphrase for expansion
def expand_query(q, n=4):
    prompts = [f"paraphrase: {q}"] * n
    outs = paraphraser(prompts, max_length=60, num_return_sequences=1)
    return [q] + [o['generated_text'] for o in outs]

# Retrieve chunks
def retrieve(q, idx, chunks):
    with st.spinner("Retrieving relevant document chunks..."): # Added spinner
        counts = {}
        for var in expand_query(q):
            v = embed_model.encode([var], convert_to_numpy=True)
            D, I = idx.search(v, 4)
            for i in I[0]:
                counts[i] = counts.get(i,0) + 1
        top = sorted(counts, key=counts.get, reverse=True)[:4]
        context = " ".join(chunks[i] for i in top)
    return context

# Generate answer
def generate_answer(q, ctx):
    prompt = f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:"
    # Added st.status for LLM generation, which is the longest step
    with st.status("Generating answer with LLM (this will take time)...", expanded=True) as status_llm:
        st.write(f"Prompting LLM with context (first 200 chars): {ctx[:200]}...") # Debugging context
        st.write(f"Question: {q}") # Debugging question

        res = llm(prompt, max_tokens=512, stop=["Question:", "\n\n"], echo=False)

        generated_text = res['choices'][0]['text'].strip()
        st.write(f"LLM Raw Response (for debugging): {res}") # Debugging raw response
        st.write(f"LLM Generated Text: {generated_text}") # Debugging final text

        status_llm.update(label="Answer Generated!", state="complete") # Removed icon parameter as per previous fix

    return generated_text

# # Streamlit UI
# st.title("Offline Doc QA with LLM")
# f = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])

# if f:
#     # Use st.session_state to persist extracted text and index across reruns
#     # This prevents re-processing the document every time the user types a question
#     if "doc_processed_file_name" not in st.session_state or st.session_state.doc_processed_file_name != f.name:
#         st.session_state.doc_processed_file_name = f.name # Store file name for new file check
        
#         st.write("Processing new document...")
#         path = "uploaded" + (".pdf" if f.type == "application/pdf" else ".docx")
#         with open(path, "wb") as o: o.write(f.getbuffer())
        
#         txt = extract_text(path)
#         st.session_state.chunks = chunk_text(txt)
#         st.session_state.index = build_index(st.session_state.chunks)
#         st.success("Document processed and index built!")

#     # Display a separate input for the question
#     q = st.text_input("Your question:", key="user_question")
    
#     # Only run the QA logic if a question is entered
#     if q:
#         # Retrieve context from the session state
#         ctx = retrieve(q, st.session_state.index, st.session_state.chunks)
        
#         # Generate answer using the context
#         ans = generate_answer(q, ctx)
        
#         st.subheader("Answer:")
#         st.write(ans)

# if f:
#     if "doc_processed_file_name" not in st.session_state or st.session_state.doc_processed_file_name != f.name:
#         st.session_state.doc_processed_file_name = f.name

#         st.write("Processing new document...")

#         # Generate a unique filename
#         file_extension = ".pdf" if f.type == "application/pdf" else ".docx"
#         unique_filename = str(uuid.uuid4()) + file_extension
#         path = os.path.join("uploaded_docs", unique_filename) # Consider creating a subfolder

#         # Ensure the directory exists (if using a subfolder like 'uploaded_docs')
#         os.makedirs(os.path.dirname(path), exist_ok=True)

#         with open(path, "wb") as o:
#             o.write(f.getbuffer())

#         txt = extract_text(path)
#         st.session_state.chunks = chunk_text(txt)
#         st.session_state.index = build_index(st.session_state.chunks)
#         st.success("Document processed and index built!")
# Streamlit UI
st.title("Offline Doc QA with LLM")
f = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])

# This is the block for processing the uploaded file.
# It only runs when a new file is selected or needs reprocessing.
if f:
    # Use st.session_state to persist extracted text and index across reruns
    if "doc_processed_file_name" not in st.session_state or st.session_state.doc_processed_file_name != f.name:
        st.session_state.doc_processed_file_name = f.name # Store file name for new file check
        
        st.write("Processing new document...")
        file_extension = ".pdf" if f.type == "application/pdf" else ".docx"
        unique_filename = str(uuid.uuid4()) + file_extension
        path = os.path.join("uploaded_docs", unique_filename) 

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as o:
            o.write(f.getbuffer())
        
        txt = extract_text(path)
        st.session_state.chunks = chunk_text(txt)
        st.session_state.index = build_index(st.session_state.chunks)
        st.success("Document processed and index built!")
        
        # It's good practice to explicitly clear the question input if a new file is loaded
        # This prevents an old question from being re-run with the new document
        if "user_question" in st.session_state:
            del st.session_state["user_question"]


# --- IMPORTANT CHANGE STARTS HERE ---

# This block will display the question input and run QA ONLY if a document
# has been successfully processed and its data is in session_state.
if "chunks" in st.session_state and "index" in st.session_state:
    st.success("Document processed and ready to answer questions!") # Or some other confirmation
    
    # Display a separate input for the question
    q = st.text_input("Your question:", key="user_question") # Moved outside the 'if f:' block
    
    # Only run the QA logic if a question is entered
    if q:
        # Retrieve context from the session state
        ctx = retrieve(q, st.session_state.index, st.session_state.chunks)
        
        # Generate answer using the context
        ans = generate_answer(q, ctx)
        
        st.subheader("Answer:")
        st.write(ans)
else:
    # Display a message if no document is uploaded or processed yet
    st.info("Please upload a document to start asking questions.")

# --- IMPORTANT CHANGE ENDS HERE ---