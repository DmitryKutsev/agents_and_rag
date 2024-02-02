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


# https://pypi.org/project/youtube-transcript-api/
# transcript = YouTubeTranscriptApi.get_transcript(youtube_ids[1], languages=['en'])
# transcript_list = YouTubeTranscriptApi.list_transcripts(youtube_ids[4])
# transcript = transcript_list.find_manually_created_transcript(['en'])
# print(transcript.fetch())

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

search_terms_prompt = PromptTemplate(
     input_variables=["text_input"],
     template="I want you to give me a single good youtube search query based on the following prompt:\n\n {text_input}"
 )

YT_create_search_terms_chain = LLMChain(llm=llm, prompt=search_terms_prompt)

# Map
map_template = """The following is a set of documents
{docs}
Based on this list of docs, please identify the main themes 
Helpful Answer:"""
map_prompt = PromptTemplate.from_template(map_template)
map_chain = LLMChain(llm=llm, prompt=map_prompt)

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

def chunk_documents (text: str) -> list:
    # doc= Document(page_content=text, metadata= {"type": "test"})
    #result = doc.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    return text_splitter.split_text(text)

    

# doc = Document(page_content= "Hello my name is Selim Berntsen, I am from Nijmegen", metadata= {"type": "test"})
# print(doc.page_content)