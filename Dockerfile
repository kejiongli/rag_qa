FROM python:3.10-slim
LABEL authors="kejiong.li"

WORKDIR /app

RUN apt-get update &&  \
    apt-get install -y build-essential curl software-properties-common git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip &&  \
#    pip install torch --index-url https://download.pytorch.org/whl/cpu &&  \
    pip install -r requirements.txt

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

COPY DB ./DB/
COPY *.py ./

ENTRYPOINT ["streamlit", "run", "run_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]