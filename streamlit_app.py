import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_google_vertexai import ChatVertexAI  # For Gemini API

# Initialize the language model (Gemini API) and prompt template
llm = ChatVertexAI(
    model_name="gemini-1.5-pro",  # Use Gemini model
    max_output_tokens=100
)

# Free embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Prompt template
str_parser = StrOutputParser()
template = (
    "Please answer the questions based on the following content and your own judgment:\n"
    "{context}\n"
    "Question: {question}"
)
prompt = ChatPromptTemplate.from_template(template)

# Streamlit App
st.title("LangChain LLM Q&A with Gemini")

# User input for the question
question = st.text_input("Ask me anything:")

# Load FAISS index
try:
    # Load pre-indexed FAISS database using free embedding model
    db_pdf = FAISS.load_local(
        "Database/PDF_All_MiniLM_L6_v2",
        embedding_model,
        allow_dangerous_deserialization=True
    )
    pdf_retriever = db_pdf.as_retriever()
    st.write("Loaded pre-indexed FAISS data successfully.")
except Exception as e:
    st.write("Error loading FAISS index:", e)
    pdf_retriever = None

# Process user input when button is clicked
if st.button("Get Answer"):
    if question and pdf_retriever:
        # Retrieve context relevant to the question
        retrieved_docs = pdf_retriever.get_relevant_documents(question)
        context_texts = "\n".join([doc.page_content for doc in retrieved_docs])

        # Format and retrieve the answer from the Gemini API
        inputs = {"context": context_texts, "question": question}
        answer = llm(prompt.format(**inputs))

        # Display the answer
        st.write("Answer:", answer.content)
    else:
        st.write("Please enter a question.")
