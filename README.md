# Llama2-7B TalkBot (Version 1.0) - Single Tenant Mode

This projects aims to create a RAG model that will fetch the data from user's private data on uploading the documents and get responses based on their personal queries.

It was developed on "Python 3.10" environment running on Linux OS (WSL will also work). Please make sure that **CUDA** and **cuDNN** drivers are enabled on your system before going further. **tensorrt** (optional) can also help to speed up inferenece time.

This project uses the following technology stack:

- **Llama2-7B-hf-chat model (from huggingface)**: This is a LLM from HuggingFace developed by Meta running on local system.

- **ChromaDB**: Open-source vector database for storing the embeddings created from documents using chroma's PersistentClient.

- **hkunlp/instructor-large (embedding model)**: This model is also from HuggingFace used for embedding the data given by the user.

- **streamlit**: For creating a simple frontend webapp to demonstrate the RAG (Retreival Augmented Generation) of LLM based on private data. (Data should be PDF format only!!) 

- **Llama-index**: For interfacing LLM and custom data sources

- **accelerate, pytorch, tensorflow**: For using GPU support on model

- **bitsandbytes**: For quantization of model to be able to run on low-resource systems

NOTE - This model requires at least 6GB VRAM for decent performance and upto 16GB RAM.

---

## Steps to install:

- pip install -r requirements.txt

- Create an access token from HuggingFace with read access to use the above models.

- Make sure to get access to Llama2 models by filling the reuqest access form on HuggingFace. This may take 1-2 hrs.

- Open a terminal and write:
    - streamlit run web_app.py
    - It will run on localhost on port 8501

---
