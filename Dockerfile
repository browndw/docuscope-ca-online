FROM python:3.11-slim

WORKDIR /webapp

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "index.py", "--server.port", "8501"]
