FROM continuumio/miniconda3

LABEL maintainer="jason@10layer.com"
LABEL version="1.0.0"
LABEL description="Document Reranking API"

WORKDIR /app

# Copy environment.yml first to leverage Docker cache
COPY environment.yml .
RUN conda env create -f environment.yml

# Activate the environment and install any additional dependencies if needed
RUN echo "source activate document-reranking-api" > ~/.bashrc
ENV PATH /opt/conda/envs/document-reranking-api/bin:$PATH

# Copy the rest of the application
COPY . .

# Create model cache directory
RUN mkdir -p model_cache

# Expose the port (this is just documentation)
EXPOSE 8000

# Set default environment variables
# ENV RERANK_HOST=0.0.0.0
# ENV RERANK_PORT=8000

# Command to run the application
CMD ["python", "api.py"] 