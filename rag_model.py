"""
    This is the documentaion of the Llama2-7B-chat model from hugging face models
    This model has 7 billion parameters develped by Meta
    
    This is used for QnA purposes on local machine for testing...
    
    Model hardware config:
        - GPU: Nvidia RTX 40 Series (12GB) --> CUDA support 
        - RAM: 32GB
        - i7 processor 13th gen
"""
import torch
from transformers import BitsAndBytesConfig
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
import chromadb

from llama_index.llms import HuggingFaceLLM
from llama_index import ServiceContext, SimpleDirectoryReader, \
    VectorStoreIndex, get_response_synthesizer, load_index_from_storage, set_global_service_context
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.prompts import PromptTemplate
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.postprocessor import SimilarityPostprocessor

from dotenv import load_dotenv
import os


load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


class Llama2_7B_Chat:
    """Class for Llama-7B Chat model from HuggingFace"""

    def __init__(self) -> None:
        """Constrcutor of the class Llama2_7B_Chat"""

        print("starting constructor...")

        # for model bit quantization for more effiency in computation by the LLM
        self.__quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        # HuggingFaceLLM object - uses pretrained models from HuggingFace (Llama2-7B-chat model)
        self.__llm = HuggingFaceLLM(
            model_name="meta-llama/Llama-2-7b-chat-hf",
            tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
            query_wrapper_prompt=PromptTemplate(
                "<s> [INST] {query_str} [/INST]"),
            context_window=4000,
            model_kwargs={
                "quantization_config": self.__quantization_config,
                "token": HF_TOKEN
            },
            tokenizer_kwargs={
                "token": HF_TOKEN
            },
            device_map="cuda"
        )

        # embedding model - pretrained embedding model (it is wrapper around sentence_transformers)
        self.__embed_model = HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-large",
            model_kwargs={
                "device": "cuda"
            }
        )

        # Query engine based on given service context
        self.__query_engine = None

        # Vector Index object
        self.__index = None

        # Service context
        self.__service_context = ServiceContext.from_defaults(
            llm=self.__llm, embed_model=self.__embed_model)

        set_global_service_context(self.__service_context)

    def create_index(self, data_dir) -> None:
        """Creates the Vector Index for querying with LLM"""

        print("creating index....")

        # Checking for existence of persistent vector_store
        if os.path.exists('vector_store_data'):
            storage_context = StorageContext.from_defaults(
                persist_dir='vector_store_data')
            self.__index = load_index_from_storage(
                storage_context=storage_context)
        else:
            # read the data from the folder
            docs = SimpleDirectoryReader(data_dir).load_data()

            # Creating persistent client
            chroma_client = chromadb.PersistentClient(path="./chroma_db")

            # create collection
            chroma_collection = chroma_client.get_or_create_collection(
                "vectorDB")

            # assign chroma store as vector_store to the context
            vector_store = ChromaVectorStore(
                chroma_collection=chroma_collection)

            storage_context = StorageContext.from_defaults(
                vector_store=vector_store)

            # creating index
            self.__index = VectorStoreIndex.from_documents(
                docs,
                storage_context=storage_context
            )

            # storing index to disk
            # self.__index.storage_context.persist(persist_dir='vector_store_data')

    def start_query_engine(self) -> None:
        """Initialize the query engine"""

        print("starting query engine...")

        # configure retriever
        retriever = VectorIndexRetriever(
            index=self.__index,
            similarity_top_k=6
        )

        # configure node postproceesors
        s_processor = SimilarityPostprocessor(similarity_cutoff=0.65)

        # configure response synthesizer
        response_synthesizer = get_response_synthesizer(
            service_context=self.__service_context)

        self.__query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[s_processor],
            response_synthesizer=response_synthesizer
        )

    def ask_llm(self, user_query: str):
        """
            Ask LLM for querying data based on context

            returns: (RESPONSE_TYPE, List[NodeWithScore])
        """

        print("User asking -->", user_query)

        response = self.__query_engine.query(user_query)
        
        print("Model says --->", response)
        
        return response, response.source_nodes


def reset_model():
    """resets the model's knowledge base"""

    os.system("rm -rf Data_*")
    os.system("rm -rf vector_store_data/")
    os.system("rm -rf chroma_db/")
