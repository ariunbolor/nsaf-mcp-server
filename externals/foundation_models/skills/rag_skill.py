from typing import Dict, Any, List, Optional
import json
from pathlib import Path
from .base_skill import BaseSkill
from ..utils.validation import validate_params
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

class RAGSkill(BaseSkill):
    """Skill for handling Retrieval-Augmented Generation (RAG) content."""

    def __init__(self):
        super().__init__()
        self.name = "rag"
        self.config = self._load_config()
        self.vector_store = None
        self.qa_chain = None

    def _load_config(self) -> Dict[str, Any]:
        """Load RAG skill configuration."""
        config_path = Path(__file__).parent / "configs" / "rag.json"
        with open(config_path, "r") as f:
            return json.load(f)

    def _initialize_rag(self, data_path: str) -> None:
        """Initialize RAG components with user data."""
        # Load documents
        loader = DirectoryLoader(data_path, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_documents(documents)

        # Create vector store
        embeddings = OpenAIEmbeddings()
        self.vector_store = FAISS.from_documents(texts, embeddings)

        # Initialize QA chain
        llm = ChatOpenAI(temperature=0.7)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever()
        )

    def _generate_content(self, query: str, context: Dict[str, Any]) -> str:
        """Generate content using RAG."""
        if not self.qa_chain:
            raise ValueError("RAG system not initialized. Please upload data first.")

        # Enhance query with context
        enhanced_query = f"""
        Generate content for: {query}
        Goal: {context.get('goal', 'general')}
        Tone: {context.get('tone', 'neutral')}
        Target Audience: {context.get('target_audience', 'general')}
        Additional Context: {context.get('additional_context', '')}
        """

        # Generate response
        response = self.qa_chain.run(enhanced_query)
        return response

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute RAG skill operations."""
        try:
            validate_params(params, self.config["params_schema"])
            action = params["action"]

            if action == "initialize":
                if "data_path" not in params:
                    raise ValueError("data_path is required for initialization")
                self._initialize_rag(params["data_path"])
                return {
                    "success": True,
                    "message": "RAG system initialized successfully"
                }

            elif action == "generate":
                if not all(k in params for k in ["query", "context"]):
                    raise ValueError("query and context are required for generation")
                content = self._generate_content(params["query"], params["context"])
                return {
                    "success": True,
                    "content": content
                }

            else:
                raise ValueError(f"Unknown action: {action}")

        except Exception as e:
            return {
                "success": False,
                "errors": [str(e)]
            }

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.vector_store:
            # Clean up vector store resources if needed
            self.vector_store = None
        if self.qa_chain:
            # Clean up QA chain resources if needed
            self.qa_chain = None
