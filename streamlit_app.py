import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import base64
import numpy as np
from component import page_style

page_style()

# --------------------------- FUNCTION TO DISPLAY PDF ---------------------------
def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# --------------------------- MODELS ---------------------------
# Text generation model (lightweight for faster response)
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Embedding model for FAISS
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Prompt template
template = (
    "Based on the following information, provide a concise answer to the question:\n\n"
    "Information:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer concisely:"
)
prompt = ChatPromptTemplate.from_template(template)

# --------------------------- MAIN INTERFACE ---------------------------
st.title("📄 Muhammad Fahmi's Resume Q&A")

st.markdown(
    """
    Welcome to the **Resume Q&A App**!  
    🔍 Ask any question about Muhammad Fahmi's professional background and skills.
    """
)

# Display PDF Resume
st.subheader("📑 Muhammad Fahmi's Resume")
display_pdf("Database/Resume/Resume_Muhammad_Fahmi_Mohd_Zainal.pdf")

# Question input
st.subheader("💬 Ask a Question")
question = st.text_input("Ask me anything about Muhammad Fahmi:", placeholder="e.g., What is his current role?")

# --------------------------- FAISS DATABASE ---------------------------
try:
    db_pdf = FAISS.load_local(
        "Database/PDF_All_MiniLM_L6_v2",
        embedding_model,
        allow_dangerous_deserialization=True
    )
    pdf_retriever = db_pdf.as_retriever()
    st.success("✅ Resume data loaded successfully!")
except Exception as e:
    st.error(f"⚠️ Error loading resume data: {e}")
    pdf_retriever = None

# --------------------------- RESPONSE GENERATION ---------------------------
if st.button("🔎 Get Answer"):
    if question and pdf_retriever:
        # Retrieve relevant documents
        retrieved_docs = pdf_retriever.get_relevant_documents(question)[:2]
        context_texts = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Generate answer
        input_prompt = prompt.format(context=context_texts, question=question)
        response = generator(input_prompt, max_length=200, do_sample=False)
        answer = response[0]["generated_text"]

        # Display the answer
        st.subheader("📌 Answer:")
        st.success(answer)
    else:
        st.warning("⚠️ Please enter a question to get started.")
