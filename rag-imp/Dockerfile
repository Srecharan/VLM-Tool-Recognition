FROM python:3.10-slim

WORKDIR /app

COPY rag_requirements.txt .
RUN pip install --no-cache-dir -r rag_requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "test_pinecone.py"] 