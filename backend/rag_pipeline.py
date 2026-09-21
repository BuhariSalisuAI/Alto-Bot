import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# 1. Janyo API Keys daga .env file (Don kiyaye sirrin ku)
load_dotenv()

def build_rag_pipeline(data_path="../data/chat_logs.txt"):
    """
    Wannan function din zai gina kwakwalwar Alto Bot ta amfani da RAG.
    """
    print("Ana karantar da Alto bayanan ku...")
    
    # 2. Loda Bayanan Group Dinku (Loading Data)
    try:
        loader = TextLoader(data_path)
        docs = loader.load()
    except FileNotFoundError:
        print(f"Error: Bamu ga fayil din {data_path} ba. Da fatan ka kirkire shi.")
        return None

    # 3. Rarraba Rubutun Don Samun Saukin Bincike (Text Splitting)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 4. Ajiye bayanan a kwakwalwar wucin gadi (Vector Database)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    # 5. Saka kwakwalwar AI da tsara yadda zata bada amsa (LLM & Prompt)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    
    system_prompt = (
        "You are Alto, a helpful AI chatbot for our team. "
        "Your job is to read our group chats, meeting summaries, and calls. "
        "Use the following retrieved context to answer the user's question accurately. "
        "If you don't know the answer based on the context, say that you don't know. "
        "Keep your answers concise and helpful."
        "\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # 6. Hada Komai Waje Daya (Chain creation)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

# Gwada Bot din (Testing the Bot)
if __name__ == "__main__":
    bot = build_rag_pipeline()
    if bot:
        question = "What is the name of our project and when is the deadline?"
        print(f"\nTambaya: {question}")
        response = bot.invoke({"input": question})
        print(f"\nAmsar Alto: {response['answer']}")
