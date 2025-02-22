import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from htmlTemplates import css, bot_template, user_template 


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    model_name = "BAAI/bge-small-en"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {"normalize_embeddings": True}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm= ChatGroq(model="llama3-8b-8192",temperature=0) 
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant answering questions based on the provided documents.
        Answer the question using only the context provided.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Keep your answers focused and relevant to the question."""),
        ("human", """Context: {context}

Question: {question}

Answer: """)
    ])

    # Create the retrieval chain using syntax
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # Define the chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

def handle_user_input(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process documents first.")
        return

    try:
        # Invoke the chain with the question
        response = st.session_state.conversation.invoke(user_question)
        
        # Update chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Add the new messages to chat history
        st.session_state.chat_history.append(("user", user_question))
        st.session_state.chat_history.append(("bot", response))

        # Display chat history
        for sender, message in st.session_state.chat_history:
            if sender == "user":
                st.write(user_template.replace("{{MSG}}", message), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message), unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"An error occurred while processing your question: {str(e)}")

def main():
    load_dotenv()
    
    # st.write(css, unsafe_allow_html=True)

    if 'user_template' not in globals():
        global user_template
        user_template = '''
        <div class="chat-message user">
            <div class="avatar">
                <img src="https://i.ibb.co/rdZC7LZ/user.png">
            </div>
            <div class="message">{{MSG}}</div>
        </div>
        '''

    if 'bot_template' not in globals():
        global bot_template
        bot_template = '''
        <div class="chat-message bot">
            <div class="avatar">
                <img src="https://i.ibb.co/cN0nmSj/robot.png">
            </div>
            <div class="message">{{MSG}}</div>
        </div>
        '''

    st.set_page_config(page_title='Chat with PDFs', page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header('PDF ChatBot 📚')
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.subheader("Upload Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click 'Process'",
            accept_multiple_files=True,
            type=['pdf']
        )
        
        if st.button('Process'):
            if not pdf_docs:
                st.warning("Please upload at least one PDF document.")
                return
                
            with st.spinner("Processing documents..."):
                try:
                    # Get PDF text
                    raw_text = get_pdf_text(pdf_docs)
                    
                    # Get text chunks
                    text_chunks = get_text_chunks(raw_text)
                    
                    # Create vector store
                    vectorstore = get_vector_store(text_chunks)
                    
                    # Create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    
                    st.success("Documents processed successfully!")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

    # Main chat interface
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_user_input(user_question)

if __name__ == "__main__":
    main()