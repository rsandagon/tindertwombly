FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

# NOTE: Run predocker for big files curl
COPY ./app /app
COPY ./requirements.txt .
RUN pip3 install -r requirements.txt --no-cache-dir
RUN pip3 install -U spacy
RUN python -m spacy download en_core_web_sm

# CMD ["bash"]