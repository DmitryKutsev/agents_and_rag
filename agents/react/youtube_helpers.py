import os
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document



llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

search_terms_prompt = PromptTemplate(
     input_variables=["text_input"],
     template="I want you to give me a single good youtube search query based on the following prompt:\n\n {text_input}"
 )

YT_create_search_terms_chain = LLMChain(llm=llm, prompt=search_terms_prompt)

final_answer_prompt = PromptTemplate(
     input_variables=["chain_output", "query"],
     template="""Use the following information from these youtube transcript summaries:
     \n\n {chain_output} \n To give answer to the following question: {query}"""
 )


create_final_answer = LLMChain(llm=llm, prompt=final_answer_prompt)

def get_youtube_video_ids(query: str, max_results: int) -> list:
    """
    Get a list of video IDs for the top n results from a YouTube search query.

    Args:
    query (str): The search term.
    max_results (int): Number of YouTube videos to process.

    Returns:
    list: List of video IDs.
    """
    
    load_dotenv()
    api_key = os.environ.get('yt_api_key')

    base_url = 'https://www.googleapis.com/youtube/v3/'
    search_url = f'{base_url}search?key={api_key}&q={query}&maxResults={max_results}&part=snippet&type=video'
    
    response = requests.get(search_url)
    data = response.json()
    items = data.get('items', [])
    
    return [item['id']['videoId'] for item in items]



def fetch_transcript(video_id: str, lang_code: str) -> str:
    """
    Get the transcript of a YouTube video in the desired language.

    Args:
    video_id (str): The ID of the YouTube video.
    lang_code (str): Desired language code for the transcript.

    Returns:
    str: The transcript of the video in the specified language.
    """
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        first_transcript_stored = False

        for transcript in transcript_list:
            if first_transcript_stored == False:
                first_transcript = transcript
                first_transcript_stored = True
            if transcript.language_code == lang_code:
                return ''.join([chunk['text'] for chunk in transcript.fetch()])
            
        if first_transcript: 
            return ''.join([chunk['text'] for chunk in first_transcript.fetch()])

        else:
            return ''
        
    except Exception as e:
        print(f"Error fetching transcript in language {lang_code} for video {video_id}: {str(e)}")
        return ''

def chunk_documents (text: str, chunk_size: int, chunk_overlap: int):
    """ 
        Get the transcript of a YouTube video in the desired language.

    Args:
    video_id (str): The ID of the YouTube video.
    lang_code (str): Desired language code for the transcript.

    Returns:
    str: The transcript of the video in the specified language.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    list_of_texts = text_splitter.split_text(text)
    return [Document(page_content = x, metadata= {"type": "test"}) for x in list_of_texts]

