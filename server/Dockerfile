# server/Dockerfile
FROM python:3.10-slim

# Install OS dependencies - unixodbc might still be pulled by pyodbc, but full MS drivers not needed for now
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg \
    unixodbc unixodbc-dev \
 && rm -rf /var/lib/apt/lists/*

# # --- MS ODBC Driver Install (Commented out for v1 SQLite/Postgres focus) ---
# RUN apt-get update && apt-get install -y curl apt-transport-https && \
#     curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
#     curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
#     apt-get update && \
#     ACCEPT_EULA=Y apt-get install -y msodbcsql18 && \
#     apt-get clean && rm -rf /var/lib/apt/lists/*
# # --- End of MS ODBC Driver Install ---

WORKDIR /app

# Copy requirements from server/requirements.txt (path from project root)
COPY ./server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code from server/app (path from project root)
COPY ./server/app /app/app/

# Copy prompts from the project root
COPY ./prompts /app/prompts/
# Copy seed_data from the project root
COPY ./rag_files/seed /app/seed_data/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
