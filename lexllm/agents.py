from qdrant_client import QdrantClient
from qdrant_client.http import models

from config import Config
from chainfury.components.tune import ChatNBX, chatnbx
import sys
from embed import query_embeddings_from_qdrant
from typing import List

DEFAULT_SUMMARIZATION_MODEL='mixtral-8x7b-inst-v0-1-32k'
DEFAULT_QA_MODEL='goliath-120b-4k-fast'
QA_SYSTEM_PROMPT = "You are given questions related to Lex Fridman's podcast videos on Youtube. Your goal is to answer those questions as accurately as possible."
SUMMARY_SYSTEM_PROMPT = "Hi there! How can I help you?"

class ChatAgent:
    def __init__(self, summarization_model: str=DEFAULT_SUMMARIZATION_MODEL, qa_model: str=DEFAULT_QA_MODEL, is_testing: bool = True) -> None:
        # create connection to Qdrant vectordb
        self.vectordb_client = self.create_vectordb_client(is_testing, vectordb_name='qdrant')

        # create OpenAI API clients
        self.summarization_model = summarization_model
        self.qa_model = qa_model
        self.is_testing = is_testing
    
    def ask(self, query: str) -> str:

        # decide what type of agent we need to answer the query

        is_video_summary_question = self.check_if_video_summary_requested(query)
        if is_video_summary_question:
            print ("The query is requesting for a video summary.")
            video_title = self.get_video_title_from_query(query)

            chapters = self.get_chapters(video_title)[0]

            return self.generate_video_summary(chapters)
        else:
            print ("The query is asking a question about a video.")
            return self.answer_question_about_video(query)
    

    def get_chapters(self, video_title):
        collection_name = Config.PROD_COLLECTION_NAME

        matches = self.vectordb_client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must = [
                    models.FieldCondition(
                        key='video_title',
                        match=models.MatchValue(value=video_title),
                    )
                ]
            ),
            with_vectors=False,
            with_payload=True
        )

        return matches

    def check_if_video_summary_requested(self, query):
        prompt = f"""
1. You are given a query about a Youtube video. Analyse the query carefully.
2. If the query's intent is to "summarize the video", then respond by 'Yes' (without the quotes)
3. In other cases, respond by 'No' (without the quotes)
4. Your response should ONLY contain 'Yes' or 'No'. No other explanatory sentences should be present. 

Example 1:
Q1: Summarize the video where Morgan Freeman appeared.
A1: Yes

Example 2:
Q2: What did Elon Musk say about Starship?
A2: No

Example 3:
Q3: Pls summarize the section in which Joe Rogan talks about self-driving cars?
A3. No

Example 4:
Q4. Pls summarize the video titled "How to be successful in life?"
A4: Yes

Example 5:
Q5. Please highlight salient points from the podcast video where Barack Obama was the guest?
A5: Yes

Example 6:
Q6: What are the views of Taylor Swift on global warming?
A6: No

Here is the question:
{query}

Answer:

"""
        response = self.invoke_chatnbx_api(prompt=prompt, model=self.summarization_model, system_prompt=SUMMARY_SYSTEM_PROMPT)

        if 'yes' in response.strip().lower():
            return True
        
        elif 'no' in response.strip().lower():
            return False
        
        else:
            raise Exception(f"Invalid response. Response can only be 'Yes'/'No'. Instead received {response}")
    
    def answer_question_about_video(self, query):
        prompt = self.add_context_to_prompt(query=query)
        print ("**"*50)
        print ("PROMPT:\n\n")
        print (prompt)
        
        answer = self.invoke_chatnbx_api(prompt, self.qa_model, QA_SYSTEM_PROMPT)

        return answer
    
    def get_video_title_from_query(self, query):
        query_results = query_embeddings_from_qdrant(query=query, 
                                                    qdrant_client=self.vectordb_client, 
                                                    qdrant_collection_name=Config.PROD_COLLECTION_NAME,
                                                    search_by='video_title',
                                                    top_k=1)
        
        extracted_video_title = query_results[0].payload['video_title'].strip('\'"').strip()

        return extracted_video_title

    def generate_video_summary(self, chapters: List) -> str:
        
        chapter_summaries = []
        for chapter in chapters:
            print (f"Summarizing chapter: {chapter.payload['title']}..")
            chapter_summaries.append(
                                    self.summarize_chapter(
                                        chapter_content=chapter.payload['content'],
                                        chapter_title=chapter.payload['title']
                                    )
                                )
        
        chapter_summaries_combined = '\n\n'.join([chapter for chapter in chapter_summaries])

        video_summary = self.summarize_video(chapter_summaries=chapter_summaries_combined)

        print ('Video summary created.')
        
        return video_summary
    
    def add_context_to_prompt(self, query: str):

        query_results = query_embeddings_from_qdrant(query=query, 
                                                         qdrant_client=self.vectordb_client, 
                                                         qdrant_collection_name=Config.PROD_COLLECTION_NAME, 
                                                         top_k=2)

        # extract context from the query response
        context = ""
        print ("Retrieved chapters...\n")
        for i, result in enumerate(query_results):
            chapter_content = result.payload['content']
            chapter_title = result.payload['title']

            print (f"{i+1}. {result.payload['title']} (Score: {round(result.score, 3)})")

            # summary = self.summarize_chapter(chapter_content=chapter_content, chapter_title=chapter_title)
            summary = self.extract_relevant_text_from_chapters(query=query, chapter_content=chapter_content, chapter_title=chapter_title)

            context += summary
            context += "\n\n"
            

        qa_prompt = f"""
1. You are given a question related to any of the Lex Fridman's podcast videos on Youtube.
2. Your goal is to answer that question as accurately as possible using ONLY the context given below.
3. The context contains salient points from the relevant sections of the podcast. You have to focus only on those points that answer the question accurately.
4. If you do not find any points that can answer the question, just output the string "I don't know" (without quotes). Do not output anything else.

QUESTION: {query}

CONTEXT: 
        
{context}

ANSWER:
            
"""

        return qa_prompt

    def extract_relevant_text_from_chapters(self, query: str, chapter_content: str, chapter_title: str):
        extraction_prompt = f"""
1. You are given a snippet from a podcast transcript. It comprises of a conversation between the guest and the host on a topic {chapter_title}.
2. Additionally, you are given a question: "{query}". 
3. Now your goal is to **ONLY** list down top 3 relevant points from the conversation that can answer the above question. You should *NOT* answer the question.
4. The output should strictly adhere to the below format.

Conversation: 

{chapter_content}

Output format:

{chapter_title}:

1.
2.
3.

"""
        
        salient_points = self.invoke_chatnbx_api(extraction_prompt, self.summarization_model, SUMMARY_SYSTEM_PROMPT)

        return salient_points

    def summarize_chapter(self, chapter_content: str, chapter_title: str):
        summarization_prompt = f"""
You are given a chapter from a podcast transcript. It comprises of a conversation between the guest and the host on a topic {chapter_title}.
Now your goal is to summarize this conversation by producing a numbered list of MAXIMUM FIVE unique, concise and accurate points about the aforementioned topic. 



Output format: 

Summary of chapter: {chapter_title}

1.
2.
3.

Conversation:

{chapter_content}

"""
        
        summary = self.invoke_chatnbx_api(summarization_prompt, self.summarization_model, SUMMARY_SYSTEM_PROMPT)
        

        return summary
    
    def summarize_video(self, chapter_summaries: str) -> str:
        summarization_prompt = f"""
1. You are given chapter wise summaries of a youtube podcast transcript.
2. Your goal is to create a well-formatted final summary for the podcast from these points.

Chapter wise summaries
**********************

{chapter_summaries}

Output format: 

Summary of the video:

"""
        
        summary = self.invoke_chatnbx_api(summarization_prompt, self.summarization_model, SUMMARY_SYSTEM_PROMPT)
        

        return summary
    
    def invoke_chatnbx_api(self, prompt: str, model: str, system_prompt: str):
        query_response = chatnbx(
            model = model,
            temperature=0.9,
            messages = [
                ChatNBX.Message(role = "system", content = "Hi there! How can I help you?"),
                ChatNBX.Message(role = "user", content = prompt),
            ]
        )

        if 'error' in query_response.keys():
            error_type = query_response['error']['type']
            error_message = query_response['error']['message']
            raise Exception(f'ChatNBX API failed due to error type: {error_type} and returned with error message: {error_message}')  
        
        return query_response['choices'][0]['message']['content'].strip('\'"').strip()


    def create_vectordb_client(self, is_testing: bool, vectordb_name: str):
        if vectordb_name == "qdrant":
            try:
                if is_testing:
                    vectordb_client = QdrantClient(
                        host='localhost',
                        port=6333
                    )

                    print ("Connecting to Qdrant local instance.")
                    response = vectordb_client.get_collection(collection_name=Config.PROD_COLLECTION_NAME)

                else:
                    vectordb_client = QdrantClient(
                        url=Config.QDRANT_DB_URL, 
                        api_key=Config.QDRANT_CLOUD_KEY,
                    )
                    
                    print ("Connecting to Qdrant cloud instance")
                    response = vectordb_client.get_collection(collection_name=Config.PROD_COLLECTION_NAME)


                # Check if the operation was successful
                if response.status == 'okay':
                    print("Client successfully connected to Qdrant database.")
                else:
                    print("Client connected, but the collection status is not optimal.")
                
                return vectordb_client

            except Exception as e:
                print(f"Failed to create or use Qdrant client: {e}", file=sys.stderr)


            





    
    