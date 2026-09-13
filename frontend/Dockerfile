FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip3 install --no-cache-dir -r requirements.txt

# HF Spaces expects 7860 (or $PORT)
EXPOSE 7860

CMD ["sh", "-c", "streamlit run app.py --server.port ${PORT:-7860} --server.address 0.0.0.0 --server.enableXsrfProtection=false"]
