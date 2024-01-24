# LexPodcast-LLM
Ask LLM anything about Lex Fridman Podcast videos on Youtube

## Setup

**1. Create virtual environment**
```
python -m venv venv
source venv/bin/activate
```

**2. Install dependencies**
```
pip install -r requirements.txt
```

**3. Setup local instance of Qdrant**
The simplest way to do this is docker. Run the following commands in the terminal
```
docker pull qdrant/qdrant
docker run -p 6333:6333 qdrant/qdrant
```

**4. Setup configuration variables in `.env` file**
1. Rename `.env.example` file to `.env`.
2. In the current version of `lexllm`, the only mandatory environment variable to specify is the `CHATNBX_KEY`.
3. `OPENAI_API_KEY` is required only when you want to create embeddings using the OpenAI Embeddings API. Since, the embeddings for each video are already provided, this can be safely skipped.
4. `QDRANT_CLOUD_KEY` and `QDRANT_DB_URL` are optional since we will be storing the embeddings in a local instance of Qdrant. I was facing some issues `(ERROR 403)` while creating a collection in Qdrant cloud DB.

**5. Store embeddings in Qdrant DB**

1. Inside `embed.py`, you can change the value of `CREATE_EMBEDDING` to `True`, if you want to recreate embeddings. 
2. Since, embeddings are already provided, I recommend to keep it to unchanged.

```
python embed.py
```

## Ask queries
1. In order to interact with the chatbot, I have provided a notebook called `chat.ipynb`.
2. It also contains some example questions that can be asked to the chatbot.
