import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader


st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

api_key=st.sidebar.text_input("Enter your Groq API Key:",type="password")

url=st.text_input("URL",label_visibility="collapsed")

llm =ChatGroq(model_name="Llama3-8b-8192", groq_api_key=api_key)

prompt_template="""
Provide a summary of the following content in 300 words:
Content:{text}

"""
prompt=PromptTemplate(template=prompt_template)

if st.button("Summarize"):
    # Validate all the inputs
    if not api_key.strip() or not url.strip():
        st.error("Please provide the information to get started")
    # Validate URL
    elif not validators.url(url):
        st.error("Please enter a valid Url. It can may be a YT video utl or website url")
    else:
        try:
            with st.spinner("Waiting..."):
                if "youtube.com" in url:
                    loader=YoutubeLoader.from_youtube_url(url,add_video_info=False)
                else:
                    #loader = UnstructuredURLLoader(urls=[url],continue_on_failure = True)
                    loader=UnstructuredURLLoader(urls=[url],ssl_verify=False,
                                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                docs=loader.load()

                chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                response=chain.run(docs)
                st.success(response)
        except Exception as e:
            st.error(f"Exception:{e}")
                    
