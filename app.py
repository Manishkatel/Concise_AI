import yt_dlp
import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.schema import Document  # Import Document schema

# Streamlit app
st.set_page_config(page_title="Langchain: Summarize Text from YT or Website", page_icon="")
st.title("Langchain: Summarize Text from YT or Website")
st.subheader("Summarize URL")

# Initialize session state variables
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = ""

if "generic_url" not in st.session_state:
    st.session_state.generic_url = ""

if "output_summary" not in st.session_state:
    st.session_state.output_summary = ""

# Sidebar for API key input
with st.sidebar:
    st.session_state.groq_api_key = st.text_input("Groq API Key", value=st.session_state.groq_api_key, type="password")

with st.container():
    # Input field in the same container
    generic_url = st.text_input("Enter the URL", value=st.session_state.generic_url)

    # Clear button in the same container
    clear_button = st.button("Clear", use_container_width=True)

# Clear button functionality
if clear_button:
    st.session_state.generic_url = ""
    st.session_state.output_summary = ""
    st.rerun()

# Summarize button below
summarize_button = st.button("Summarize the Content", use_container_width=True)

# Ensure API key and URL are provided before proceeding
if summarize_button:
    if not st.session_state.groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the API key and URL to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL (YouTube or website)")
    else:
        try:
            with st.spinner("Processing..."):    # loader
                if "youtube.com" in generic_url:
                    # Use yt-dlp to fetch video details
                    ydl_opts = {
                        'quiet': True,
                        'forcejson': True,
                        'extract_flat': True,  # This ensures that we just get the video info
                    }
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info_dict = ydl.extract_info(generic_url, download=False)
                        # Extract video title and description
                        video_title = info_dict.get('title', 'No title available')
                        video_description = info_dict.get('description', 'No description available')

                    # Convert to LangChain document format
                    docs = [Document(page_content=video_title + "\n" + video_description)]

                else:
                    # Load content from the website using UnstructuredURLLoader
                    loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify=False, headers={})
                    docs = loader.load()

                # Initialize LLM
                llm = ChatGroq(model="mixtral-8x7b-32768", groq_api_key=st.session_state.groq_api_key)

                # Define the prompt template with prompt
                prompt_template = """
                Provide a summary of the following content in 300 words:
                Content: {text}
                """
                prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

                # Summarization chain
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                st.session_state.output_summary = chain.run(docs)

                st.success(st.session_state.output_summary)
        except Exception as e:
            st.error(f"Exception: {e}")

# Display summary output
if st.session_state.output_summary:
    st.subheader("Summary:")
    st.write(st.session_state.output_summary)
