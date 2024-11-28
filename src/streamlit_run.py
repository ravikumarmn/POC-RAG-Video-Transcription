import streamlit as st
from streamlit_player import st_player
import re
from youtube_transcript_api import YouTubeTranscriptApi
import json
from typing import List, Tuple, Dict
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

from db_utils import TranscriptDB

# Initialize database
db = TranscriptDB()

# Sample videos
SAMPLE_VIDEOS = [
    {
        "title": "Introduction to Report Generation",
        "url": "https://www.youtube.com/watch?v=3jnViQZKYHE",
    },
    {
        "title": "LlamaIndex Workshop: Building RAG with Knowledge Graphs",
        "url": "https://www.youtube.com/watch?v=hb8uT-VBEwQ",
    },
    {
        "title": "Research Paper Report Generating Agent",
        "url": "https://www.youtube.com/watch?v=8MbhRGbKZx8",
    },
    {
        "title": "LlamaIndex Webinar: Agents Showcase!",
        "url": "https://www.youtube.com/watch?v=C5NhoMBkaQU",
    },
    {
        "title": "LangChain vs LangGraph: A Tale of Two Frameworks",
        "url": "https://www.youtube.com/watch?v=qAF1NjEVHhY",
    },
]


class VideoTranscriptRAG:
    def __init__(self, transcript_data: Dict):
        """Initialize the RAG system with transcript data"""
        self.transcript_data = transcript_data
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.initialize_vector_store()

    def initialize_vector_store(self):
        """Initialize vector store with transcript segments"""
        texts = []
        metadatas = []

        # Process each transcript segment
        for segment in self.transcript_data['transcript']:
            texts.append(segment['text'])
            metadatas.append({
                'start_time': segment['start'],
                'end_time': segment['end'],
                'start_formatted': segment['start_formatted'],
                'end_formatted': segment['end_formatted'],
                'text': segment['text']
            })

        # Create vector store
        self.vector_store = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )

    def query_transcript(self, query: str) -> Dict:
        """Query the transcript and return answer with time range"""
        # Create retriever
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # Create prompt template
        PROMPT = PromptTemplate(
            template="""You are an AI assistant analyzing a video transcript. Use the provided context to answer the question.

Context from transcript:
{context}

Question: {question}

Provide a clear and detailed answer based on the context. Focus on accuracy and completeness.

Answer: """,
            input_variables=["context", "question"]
        )

        # Create QA chain
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0.3, model="gpt-4o-mini", api_key=st.secrets.get("OPENAI_API_KEY", None)),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": PROMPT,
                "verbose": True
            }
        )

        # Get response
        response = qa({"query": query})
        
        # Get time range from source documents
        source_docs = response['source_documents']
        if source_docs:
            # Sort by start time
            sorted_docs = sorted(source_docs, key=lambda x: x.metadata['start_time'])
            time_range = {
                'start_time': round(sorted_docs[0].metadata['start_time'], 2),
                'end_time': round(sorted_docs[-1].metadata['end_time'], 2)
            }
        else:
            time_range = {'start_time': None, 'end_time': None}

        return {
            'answer': response['result'],
            'time_range': time_range
        }

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    patterns = [
        r"(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_transcript(video_id):
    """Get transcript from YouTube and format it"""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

        # Format transcript data
        transcript_data = {
            "video_id": video_id,
            "metadata": {"url": f"https://www.youtube.com/watch?v={video_id}"},
            "transcript": [],
        }

        for item in transcript_list:
            # Format timestamps
            start_seconds = item["start"]
            duration = item["duration"]
            end_seconds = start_seconds + duration

            start_formatted = f"{int(start_seconds//3600):02d}:{int((start_seconds%3600)//60):02d}:{int(start_seconds%60):02d}"
            end_formatted = f"{int(end_seconds//3600):02d}:{int((end_seconds%3600)//60):02d}:{int(end_seconds%60):02d}"

            transcript_data["transcript"].append(
                {
                    "start": start_seconds,
                    "end": end_seconds,
                    "start_formatted": start_formatted,
                    "end_formatted": end_formatted,
                    "text": item["text"],
                    "duration": duration,
                }
            )

        return transcript_data
    except Exception as e:
        st.error(f"Error getting transcript: {str(e)}")
        return None


def process_video(video_id, title="", url=""):
    """Process video transcript and store in database"""
    # Check if transcript exists in database
    transcript_data = db.get_transcript(video_id)

    if not transcript_data:
        with st.spinner("Extracting transcript..."):
            transcript_data = get_transcript(video_id)
            if transcript_data:
                # Store in database
                if db.upsert_transcript(video_id, title, url, transcript_data):
                    st.success("Transcript extracted and saved!")
            else:
                st.error("Failed to extract transcript")
                return None

    # Display transcript data in sidebar
    st.sidebar.markdown("---")

    if transcript_data:
        st.success("Transcript loaded!")
        st.sidebar.markdown("### Transcript Data:")
        st.sidebar.json(transcript_data, expanded=False)

        # Extract and display only transcript text
        st.sidebar.markdown("### Transcript Text:")
        transcript_text = " ".join(
            [segment["text"] for segment in transcript_data["transcript"]]
        )
        st.sidebar.markdown(
            f"<div style='max-height: 300px; overflow-y: auto'>{transcript_text}</div>",
            unsafe_allow_html=True,
        )

    return transcript_data


def seconds_to_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def timestamp_to_seconds(timestamp):
    """Convert HH:MM:SS to seconds"""
    try:
        h, m, s = map(int, timestamp.split(":"))
        return h * 3600 + m * 60 + s
    except:
        return 0


def main():
    st.title("Video Transcript Query System")

    # Sidebar with sample videos
    st.sidebar.title("Sample Videos")
    selected_video = None

    # Display sample videos with extracted IDs
    st.sidebar.markdown("### Available Videos:")
    for video in SAMPLE_VIDEOS:
        video_id = extract_video_id(video["url"])
        if st.sidebar.button(f"{video['title']}"):
            selected_video = {**video, "video_id": video_id}

    # Custom URL input
    st.sidebar.markdown("---")
    st.sidebar.subheader("Or enter custom URL:")
    youtube_url = st.sidebar.text_input("YouTube URL:")
    if youtube_url:
        custom_video_id = extract_video_id(youtube_url)
        if custom_video_id:
            st.sidebar.markdown(f"**Extracted Video ID:** {custom_video_id}")
        else:
            st.sidebar.error("Invalid YouTube URL")

    # Process selected or custom video
    video_id = None
    if selected_video:
        video_id = selected_video["video_id"]
        youtube_url = selected_video["url"]
    elif youtube_url:
        video_id = extract_video_id(youtube_url)
        if not video_id:
            st.error("Invalid YouTube URL")
            return

    if video_id:
        # Process video and get transcript
        title = selected_video["title"] if selected_video else ""
        transcript_data = process_video(video_id, title, youtube_url)

        if transcript_data:
            # Initialize RAG system with transcript data
            rag = VideoTranscriptRAG(transcript_data)

            # Query input
            query = st.text_input("Enter your query about the video:")

            if query:
                with st.spinner("Processing query..."):
                    result = rag.query_transcript(query)

                    # Display answer
                    st.subheader("Answer:")
                    st.write(result["answer"])

                    # Get start and end times from the response
                    start_time = result["time_range"]["start_time"]
                    end_time = result["time_range"]["end_time"]

                    if start_time is not None and end_time is not None:
                        col1, col2 = st.columns(2)

                        with col1:
                            start_timestamp = seconds_to_timestamp(start_time)
                            input_start = st.text_input(
                                "Adjust start time (HH:MM:SS):", value=start_timestamp
                            )
                            start_time = timestamp_to_seconds(input_start)

                        with col2:
                            end_timestamp = seconds_to_timestamp(end_time)
                            input_end = st.text_input(
                                "Adjust end time (HH:MM:SS):", value=end_timestamp
                            )
                            end_time = timestamp_to_seconds(input_end)

                        # Create YouTube URL with start and end times
                        video_url = f"https://www.youtube.com/embed/{video_id}?start={int(start_time)}&end={int(end_time)}"

                        # Display video player
                        st.subheader("Video at Timestamp:")
                        st_player(video_url, playing=True, loop=True)

                        # Display time range information
                        st.info(
                            f"Playing video segment from {seconds_to_timestamp(start_time)} to {seconds_to_timestamp(end_time)}"
                        )


if __name__ == "__main__":
    main()
