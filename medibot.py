import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq


## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

DB_FAISS_PATH="vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt
    

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN, "max_length":"512"}
    )
    return llm

def format_sources(source_documents):
    """Format source documents nicely"""
    if not source_documents:
        return ""
    
    sources = []
    for i, doc in enumerate(source_documents[:3]):  # Show top 3 sources
        # Extract page number if available
        page = doc.metadata.get('page', 'Unknown')
        source_name = doc.metadata.get('source', 'Medical Encyclopedia')
        
        # Get filename only (remove path)
        if '/' in source_name:
            source_name = source_name.split('/')[-1]
        
        sources.append(f"📄 {source_name} (Page {page})")
    
    return "\n\n**📚 Sources:**\n" + "\n".join(sources)

def main():
    st.title("🏥 Medical AI Assistant")
    
    # Add a subtitle
    st.markdown("*Ask me any medical question and I'll provide evidence-based answers*")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])
    
    prompt=st.chat_input("Ask your medical question here...")
    
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})
        
        # Improved prompt template for better responses
        CUSTOM_PROMPT_TEMPLATE = """
You are a knowledgeable medical AI assistant. Use the provided medical information to answer the user's question clearly and professionally.

INSTRUCTIONS:
- Provide accurate, well-structured medical information
- Use clear, professional language
- Break down complex concepts when needed
- If you don't know something from the context, say so honestly
- Do not provide personal medical advice or diagnosis
- Always remind users to consult healthcare professionals for serious concerns

Context: {context}

Question: {question}

Answer:"""

        try:
            with st.spinner("🔍 Searching medical database..."):
                vectorstore=get_vectorstore()
                if vectorstore is None:
                    st.error("Failed to load the vector store")
                    return
                
                qa_chain = RetrievalQA.from_chain_type(
                    llm=ChatGroq(
                        model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
                        temperature=0.1,  # Lower temperature for more factual responses
                        groq_api_key=os.environ["GROQ_API_KEY"],
                    ),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )
                
                response=qa_chain.invoke({'query':prompt})
                result=response["result"]
                source_documents=response["source_documents"]
                
                # Clean formatting - MAIN IMPROVEMENT HERE
                clean_result = result.strip()
                
                # Add medical disclaimer
                disclaimer = "\n\n⚠️ **Disclaimer:** This information is for educational purposes only. Always consult with healthcare professionals for medical advice."
                
                # Format sources nicely
                formatted_sources = format_sources(source_documents)
                
                # Combine everything with clean formatting
                final_response = clean_result + formatted_sources + disclaimer
                
                st.chat_message('assistant').markdown(final_response)
                st.session_state.messages.append({'role':'assistant', 'content': final_response})
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("💡 Make sure your Groq API key is properly set in the .env file")

if __name__ == "__main__":
    main()