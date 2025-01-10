import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import base64

# --------------------------- PAGE CONFIGURATION ---------------------------
st.set_page_config(
    page_title="Resume Q&A | Muhammad Fahmi",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------- FUNCTION TO DISPLAY PDF ---------------------------
def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# --------------------------- SIDEBAR ---------------------------
with st.sidebar:
    st.title("📚 How to Use This App")
    st.markdown("""
    This is a **Retrieval-Augmented Generation (RAG)** app that allows you to ask questions about **Muhammad Fahmi's Resume**.

    ### 🤖 Models Used:
    - **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (for semantic search)
    - **Text Generation Model:** `google/flan-t5-small` (for generating answers)

    ### 🗂️ Database:
    - The resume data is stored in a **FAISS** vector database for fast document retrieval.

    ### 💡 Example Questions:
    - "What is Muhammad Fahmi's current job?"
    - "List Fahmi's technical skills."
    - "Tell me about his data science experience."

    ---
    """)
    st.info("⚠️ For the best results, ask clear and specific questions.")

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
display_pdf("/Database/Resume/Resume_Muhammad_Fahmi_Mohd_Zainal.pdf")

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
