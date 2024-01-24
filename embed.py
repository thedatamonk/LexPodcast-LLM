from typing import List
from qdrant_client.http import models
from openai import OpenAI
import os
import json
import tiktoken
import pandas as pd
from ast import literal_eval
from config import Config
from qdrant_client import QdrantClient
import sys

EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def load_json(json_path: str):
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    return data

def text_to_embedding(chunk: str):
    
    # initialize OpenAI client
    client = OpenAI()

    # generate embeddings using OpenAI Embeddings endpoint
    response = client.embeddings.create(
        input=chunk,
        model="text-embedding-ada-002"
    )

    return response.data[0].embedding

import tiktoken

def truncate_text_tokens(text, encoding_name=EMBEDDING_ENCODING, max_tokens=EMBEDDING_CTX_LENGTH):
    """Truncate a string to have `max_tokens` according to the given encoding."""
    encoding = tiktoken.get_encoding(encoding_name)
    return encoding.encode(text)[:max_tokens]

def create_embeddings(content: str):

    # first check how many tokens does the content correspond to
    num_tokens = num_tokens_from_string(content)
    if num_tokens > EMBEDDING_CTX_LENGTH:
        print (f"Input string contain {num_tokens} tokens that exceeds token limit of {EMBEDDING_CTX_LENGTH}.")
        print ("Truncating the string..")
        truncated_text = truncate_text_tokens(text=content)
        return text_to_embedding(truncated_text)
    
    embedding = text_to_embedding(chunk=content)

    return embedding
        
def embed_transcript(transcript_file: str) -> List:

    # read transcript file and split into chapters
    chapters = {}
    chapter_titles = []
    current_title = None
    current_content = []
    
    CHAPTER_DELIMITER = "**"*50

    with open(transcript_file, "r") as file:
        content = file.readlines()


    for line in content:

        # check if the line is a chapter delimiter
        if line.strip() == CHAPTER_DELIMITER and current_title is not None:

            # save the current chapter content
            chapters[current_title] = '\n'.join(current_content).strip()
            current_content = []
            current_title = None

        elif line.strip().startswith('CHAPTER:') and current_title is None:
            # Extract chapter title
            current_content.append(line.strip())

            current_title = line.strip().split('CHAPTER:')[1].strip()
            chapter_titles.append(current_title)

        elif current_title is not None:
            # Add line to current chapter content
            current_content.append(line.strip())
    
    if current_title and current_content:
        chapters[current_title] = '\n'.join(current_content).strip()


    # create embeddings for each chapter
    transcript_with_embeddings = []

    for title, content in chapters.items():
        content_embedding = create_embeddings(content)
        chapter_title_embedding = create_embeddings(title)

        transcript_with_embeddings.append(
            {
                "title": title,
                "content": content,
                "chapter_title_embedding": chapter_title_embedding,
                "content_embedding": content_embedding,
            }
        )

    
    return transcript_with_embeddings
   


def create_qdrant_points_data_for_video(video_folder_path):
    print (f'Processing {video_folder_path}...')
    transcript_path = os.path.join(video_folder_path, "transcript.txt")
    metadata_path = os.path.join(video_folder_path, "metadata.json")
    qdrant_data_path = os.path.join(video_folder_path, "qdrant_data.csv")

    transcript_with_embeddings = embed_transcript(transcript_file=transcript_path)

    qdrant_points_df = pd.DataFrame(transcript_with_embeddings)

    metadata = load_json(metadata_path)

    qdrant_points_df['video_title'] = metadata['title']
    qdrant_points_df['video_title_embedding'] = str(text_to_embedding(chunk=metadata['title']))
    qdrant_points_df['lex_podcast_guest_name'] = metadata['title'].split(':')[0]
    qdrant_points_df['video_url'] = metadata['video_url']
    qdrant_points_df['yt_video_id'] = metadata['video_id']

    qdrant_points_df.to_csv(qdrant_data_path, index=False)
    print (f"Qdrant data for video saved in {qdrant_data_path}")


def save_embeddings_in_qdrant(qdrant_client, qdrant_file_path, video_id, qdrant_collection_name='Test_Lex_Fridman_Podcast'):
    
    # read csv file
    qdrant_data = pd.read_csv(qdrant_file_path)

    qdrant_data["video_title_embedding"] = qdrant_data.video_title_embedding.apply(literal_eval)
    qdrant_data["chapter_title_embedding"] = qdrant_data.chapter_title_embedding.apply(literal_eval)
    qdrant_data["content_embedding"] = qdrant_data.content_embedding.apply(literal_eval)

    
    # save vectors and payload
    qdrant_client.upsert(
        collection_name=qdrant_collection_name,
        points=[
            models.PointStruct(
                id=literal_eval(str(video_id) + str(index)),
                vector={
                    "video_title": row["video_title_embedding"],
                    "chapter_title": row["chapter_title_embedding"],
                    "content": row["content_embedding"]
                },
                payload={col_name: col_val for col_name, col_val in row.to_dict().items() if col_name not in ['video_title_embedding', 'chapter_title_embedding', 'content_embedding']}
            )
            for index, row in qdrant_data.iterrows()
        ],
    )

    print (f"Embeddings for qdrant file {qdrant_file_path} saved in DB.")



def query_embeddings_from_qdrant(query, qdrant_client, qdrant_collection_name='Test_Lex_Fridman_Podcast', search_by='content', top_k=3):

    query_embedding = create_embeddings(content=query)

    query_results = qdrant_client.search(
        collection_name = qdrant_collection_name,
        query_vector = (
            search_by, query_embedding
        ),
        limit = top_k
    )

    return query_results


def create_vectordb_client(is_testing: bool, vectordb_name: str):
    if vectordb_name == "qdrant":
        try:
            if is_testing:
                vectordb_client = QdrantClient(
                    host='localhost',
                    port=6333
                )

                print ("Connecting to Qdrant local instance...")

            else:
                vectordb_client = QdrantClient(
                    url=Config.QDRANT_DB_URL, 
                    api_key=Config.QDRANT_CLOUD_KEY,
                )
                
                print ("Connecting to Qdrant cloud instance...")

            print ('Connected successfully.')
            return vectordb_client

        except Exception as e:
            print(f"Failed to create or use Qdrant client: {e}", file=sys.stderr)


if __name__ == "__main__":

    # create qdrant client in qdrant cloud
    qdrant_client = create_vectordb_client(is_testing=True, vectordb_name='qdrant')
    
    # create collection in qdrant cloud
    vector_size = 1536

    CREATE_EMBEDDING = False

    qdrant_client.create_collection(
        collection_name=Config.PROD_COLLECTION_NAME,
        vectors_config={
            "video_title": models.VectorParams(
                distance=models.Distance.COSINE,
                size=vector_size,
            ),
            "chapter_title": models.VectorParams(
                distance=models.Distance.COSINE,
                size=vector_size,
            ),
            "content": models.VectorParams(
                distance=models.Distance.COSINE,
                size=vector_size,
            ),
        }
    )

    
    # Iterate over all the transcript folders
    BASE_DIR = 'lex_transcripts'

    if CREATE_EMBEDDING:
        for folder in sorted(os.listdir(BASE_DIR)):
            path = os.path.join(BASE_DIR, folder)
            
            if os.path.isdir(path):
                create_qdrant_points_data_for_video(video_folder_path=path)

    # for each transcript folder call create_qdrant_points_data_for_video() method
    unique_video_id = 100
    for folder in sorted(os.listdir(BASE_DIR)):
        qdrant_file_path = os.path.join(BASE_DIR, folder, "qdrant_data.csv")

        if os.path.exists(qdrant_file_path):
            save_embeddings_in_qdrant(qdrant_client=qdrant_client, 
                                      qdrant_file_path=qdrant_file_path, 
                                      video_id=unique_video_id, 
                                      qdrant_collection_name=Config.PROD_COLLECTION_NAME)

            unique_video_id += 100

    


