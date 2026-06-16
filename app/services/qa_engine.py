from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from services.vector_store import get_vector_store
from os import getenv
from dotenv import load_dotenv

load_dotenv()

def ask_question(question: str)->dict:
    """Searches the vector db and uses chatgpt llm to answer the question"""

    vector_store = get_vector_store()

    retriever = vector_store.as_retriever(search_kwargs={'k':4})

    llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0, api_key=getenv('GEMINI_API_KEY'))

    template = """Answer the question based on the following context:
    {context}

    Question:{question}

    If answer is not in context, say "I don't know based on provided document"
    """

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return '\n\n'.join(doc.page_content for doc in docs)
    
    #The langchain pipeline
    rag_chain = (
        {'context': retriever | format_docs, 'question': RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(question)

    sources_docs = retriever.invoke(question)

    sources = [{'source': doc.metadata['source'],'chunk': doc.page_content} for doc in sources_docs]

    return{'answer': answer, 'sources': sources}