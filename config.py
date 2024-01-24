# config.py
from dotenv import load_dotenv
import os

class Config:
    load_dotenv()

    QDRANT_CLOUD_KEY = os.getenv('QDRANT_CLOUD_KEYQDRANT_CLOUD_KEY')
    QDRANT_DB_URL = os.getenv('QDRANT_DB_URL')

    TEST_COLLECTION_NAME = os.getenv('TEST_COLLECTION_NAME')
    PROD_COLLECTION_NAME = os.getenv('PROD_COLLECTION_NAME')

    os.environ['CHATNBX_KEY'] = os.getenv('CHATNBX_KEY')
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


    
