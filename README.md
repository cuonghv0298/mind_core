# DEPLOY MIND

## Brief

## How to run

### Install the dependencies

You can advisably create a virtual environment

```bash
conda create -n deploy_mind python=3.9
pip install -r requirements.txt
```

### *Dev env*
```
Create .env file in api folder with content:
OPENAI_API_KEY= '' # input your key
LANGCHAIN_API_KEY="" #Key from langsmith
WEAVIATE_HOST = "10.100.224.34" # Based on your resource
WEAVIATE_HOST_PORT = "8080"
WEAVIATE_GPC_URL = "10.100.224.34"
WEAVIATE_GPC_URL_PORT = "50051"
REDIS_HOST = "10.100.224.34"
REDIS_PORT = "6379"
``` 
### *Install vector embedding DB (weaviate)*
```Bash
cd src/driver/weaviatedocker
docker compose up -d
```

### *Install history_chat (redis)*
1. Pull and start the container:
```Bash
docker pull redis/redis-stack-server
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack-server:latest
```
2. - If redis-stack exited, you need to stop and remove it, then start it again:
```Bash
docker stop redis-stack
docker rm redis-stack
```
3. - (Optional) Access Redis CLI or bash:
```Bash
docker exec -it redis-stack redis-cli
# or
docker exec -it redis-stack bash
```

For more details, refer to [Redis documentation](https://redis.io/docs/)

### *Install [Ollama](https://github.com/ollama/ollama/blob/main/README.md) for the local*
1. We install with ubuntu server with curl command ([Download](https://ollama.com/download/linux)):
```Bash
curl -fsSL https://ollama.com/install.sh | sh
```
2. Pull model from ollama:
```Bash
ollama pull phi
```
### Run the app

```bash
$ streamlit run src/Home.py --server.port 8087
```

## Ref
1. Icon list: https://gist.github.com/rxaviers/7360908