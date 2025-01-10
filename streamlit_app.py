import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Use a lightweight, non-gated model for faster performance
model_name = "HuggingFaceTB/SmolLM-135M-Instruct"  # Lightweight and efficient
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Optimized embedding model for FAISS
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Prompt template
str_parser = StrOutputParser()
template = (
    "Please answer the questions based on the following content and your own judgment:\n"
    "{context}\n"
    "Question: {question}"
)
prompt = ChatPromptTemplate.from_template(template)

# Streamlit App
st.title("LangChain LLM Q&A with Free Model")

# User input for the question
question = st.text_input("Ask me anything:")

# Load FAISS index
try:
    db_pdf = FAISS.load_local(
        "Database/PDF_All_MiniLM_L6_v2",
        embedding_model,
        allow_dangerous_deserialization=True
    )
    pdf_retriever = db_pdf.as_retriever()
    st.write("Loaded pre-indexed FAISS data successfully.")
except Exception as e:
    st.error(f"Error loading FAISS index: {e}")
    pdf_retriever = None

# Process user input when button is clicked
if st.button("Get Answer"):
    if question and pdf_retriever:
        # Limit the number of retrieved documents to reduce context size
        retrieved_docs = pdf_retriever.get_relevant_documents(question)[:3]  # Limit to 3 docs
        context_texts = "\n".join([doc.page_content[:500] for doc in retrieved_docs])  # Limit each doc to 500 chars

        # Format the prompt
        input_prompt = prompt.format(context=context_texts, question=question)

        # Tokenize with truncation to avoid overflow
        inputs = tokenizer(
            input_prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024  # Reduced for smaller models
        )

        # Generate answer
        response = model.generate(
            **inputs,
            max_new_tokens=150,  # Reduced to fit smaller model capacity
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

        answer = tokenizer.decode(response[0], skip_special_tokens=True)
        st.write("Answer:", answer)
    else:
        st.warning("Please enter a question.")
