import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
    

 
load_dotenv()
 
def main():
    page_bg_img = """
    <style>
    [data-testid = "stAppViewContainer"]
    {
        background-color: #e5e5f7;
        opacity: 0.9;
        background: repeating-linear-gradient( 45deg, #E2D870, #E2D870 5px, #e5e5f7 5px, #e5e5f7 25px );
    }
    [data-testid = "stHeader"]
    {
        background-color : rgba(0,0,0,0);
    }
    </style>
    """ 
    st.markdown(page_bg_img,unsafe_allow_html = True)
    st.markdown(f'<h1 style="color:#ff0000;font-size:48px;">{"Chat with PDF 💬"}</h1>', unsafe_allow_html=True)
 
    add_vertical_space(5)
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF 👇🏽 ", type='pdf')
    
 
 
    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
 
        # # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        # st.write(chunks)
 
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')s
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
 
        # embeddings = OpenAIEmbeddings()
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
 
        # Accept user questions/query
        query = st.text_input("HIT any questions in context of the uploaded PDF")
        # st.write(query)
 
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
 
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)
            
    st.markdown(f'<h1 style="color:#ff0000;font-size:48px;">{"Developed by Sai Dinesh : )"}</h1>', unsafe_allow_html=True)
 
if __name__ == '__main__':
    main()