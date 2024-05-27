import streamlit as st
from langchain_community.llms import Ollama
import requests
from bs4 import BeautifulSoup
import PyPDF2
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import ollama
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

MODEL = "Llama3"

st.title('LLM-RAG-WebUI-integration!!!!')

#Toggle Functionality to toggle to RAG mode
activated = st.toggle("RAG Mode")
#LLM model and embedding to generate responses
embeddings = OllamaEmbeddings()
model = Ollama(model=MODEL)


rag_template = """
 You are a chatbot having a friendly conversation with a human. Based on the provided context generate assistant's response.
 context:{context}
  ******************
{chat_history}
{human_message} 
 assistant: 
 ****************** 
**Only generate assistant's response**
 """

retrieval_qa_chat_prompt = PromptTemplate(input_variables = ['context','chat_history',"human_message"], template = rag_template)


#template and prompt for the llm model
template = """
 You are a chatbot having a friendly conversation with a human.
 {chat_history} 
 'role': 'user', 'content': {Human_input}
 'role': 'assistant', 'content': 
**Only generate assistant's response**
 """

prompt = PromptTemplate(input_variables = ['chat_history'], template = template)


#Using LLM  chian to invoke responses
chain = LLMChain(
    llm=model,
    prompt=prompt
)



#Writing some helper functions
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def Ai_response(prompt, chat_history):
    result = chain.invoke(input = {
    "Human_input":prompt, "chat_history":chat_history
    })
    ans = result['text']
    return ans

def display_human(prompt):
    with st.chat_message("user"):
        st.write(prompt)
    return None

def display_llm(response):
    with st.chat_message("assistant"):
        st.write(response)
    return None

def remove_quotes(input_string):
    if input_string.startswith('"') or input_string.startswith("'"):
        input_string = input_string[1:]

    if input_string.endswith('"') or input_string.endswith("'"):
        input_string = input_string[:-1]

    return input_string

#If RAG mode is switched on this will display the container to take pdf and blog links as input
if activated:
    link = st.text_input("Enter the link here!!!!")
    st.text("Or")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

rag_messages =[]
messages = []

def main():

    if "messages" not in st.session_state:
        st.session_state.messages = []


    for message in st.session_state.messages:
        with st.chat_message(message.get("role")):
            st.write(message.get("content"))


    prompt = st.chat_input("How can I help you today!!!")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_human(prompt)

        if activated:
            if link:
                response = requests.get(link)
                soup = BeautifulSoup(response.content, "html.parser")
                Blog_text = soup.get_text()
                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=2000,
                    chunk_overlap=200,
                    length_function=len,
                )
                texts = text_splitter.split_text(Blog_text)
                document_search = FAISS.from_texts(texts, embeddings)

                ret = document_search.as_retriever()

                combine_docs_chain = create_stuff_documents_chain(
                    model, retrieval_qa_chat_prompt
                )
                retrieval_chain = create_retrieval_chain(ret, combine_docs_chain)

                ans = retrieval_chain.invoke(
                    {"input": prompt, "chat_history": st.session_state.messages, "human_message":prompt})

                response = remove_quotes(ans["answer"])
                st.session_state.messages.append({"role": "assistant", "content": response})
                display_llm(response)


            elif uploaded_file:
                raw_text = read_pdf(uploaded_file)
                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=2000,
                    chunk_overlap=200,
                    length_function=len,
                )
                texts = text_splitter.split_text(raw_text)

                document_search = FAISS.from_texts(texts, embeddings)

                ret = document_search.as_retriever()

                combine_docs_chain = create_stuff_documents_chain(
                    model, retrieval_qa_chat_prompt
                )
                retrieval_chain = create_retrieval_chain(ret, combine_docs_chain)

                ans = retrieval_chain.invoke(
                    {"input": prompt, "chat_history": st.session_state.messages, "human_message":prompt})

                response = remove_quotes(ans["answer"])
                st.session_state.messages.append({"role": "assistant", "content": response})
                display_llm(response)

        else:
            messages.append({"role": "user", "content": prompt})
            print(messages)
            result = remove_quotes(Ai_response(prompt, st.session_state.messages))
            st.session_state.messages.append({"role": "assistant", "content": result})
            with st.chat_message("assistant"):
                st.write(result)






if __name__ == "__main__":
    main()

