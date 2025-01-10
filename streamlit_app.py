import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from PIL import Image
from component import page_style

# Apply custom styles from the component
page_style()

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

# --------------------------- DISPLAY RESUME IMAGES IN TWO COLUMNS ---------------------------
st.subheader("📑 Muhammad Fahmi's Resume")

# Create two columns for the two resume pages
col1, col2 = st.columns(2)

# Column 1: Resume Page 1
with col1:
    with st.expander("📄 **Show/Hide Resume - Page 1**"):
        image1 = Image.open("Database/Resume/Resume_page1.png")
        st.image(image1, caption="Resume - Page 1", use_container_width=True)

# Column 2: Resume Page 2
with col2:
    with st.expander("📄 **Show/Hide Resume - Page 2**"):
        image2 = Image.open("Database/Resume/Resume_page2.png")
        st.image(image2, caption="Resume - Page 2", use_container_width=True)

# --------------------------- QUESTION INPUT ---------------------------
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
