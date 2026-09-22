import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# 1. Loda Mabuɗin Sirri
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def initialize_rag_pipeline(chat_log_path="data/chat_logs.txt"):
    """
    Wannan function din yana karanta bayanan tattaunawa, 
    ya raba su, kuma ya kirkiri kwakwalwar RAG.
    """
    # 2. Karanta bayanan
    if not os.path.exists(chat_log_path):
        return None, "Chat log file not found."
    
    loader = TextLoader(chat_log_path)
    documents = loader.load()

    # 3. Raba bayanan zuwa guntaye (Chunks)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # 4. Mayar da bayanan zuwa Vectors kuma adana su a ChromaDB
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = Chroma.from_documents(texts, embeddings)

    # 5. Kirkirar sarkar tattaunawa (Conversational Chain)
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", openai_api_key=api_key)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    
    return qa_chain, "Success"

def ask_alto(qa_chain, question):
    """
    Function domin tura tambaya ga Bot din
    """
    if not qa_chain:
        return "RAG pipeline not initialized properly."
    
    response = qa_chain({"question": question})
    return response['answer']

# Misalin Yadda Zai Yi Aiki
if __name__ == "__main__":
    print("⏳ Alto-Bot yana hada kwakwalwarsa...")
    chain, status = initialize_rag_pipeline()
    
    if status == "Success":
        test_question = "What is Keith doing in this Hackathon?"
        print(f"👤 Tambaya: {test_question}")
        answer = ask_alto(chain, test_question)
        print(f"🤖 Alto-Bot: {answer}")
    else:
        print(f"🛑 Error: {status}")
