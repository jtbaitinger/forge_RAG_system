"""FastAPI application for Wikipedia RAG system."""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from data_loader import WikipediaDataLoader, load_wikipedia_documents
from embedding_generator import EmbeddingGenerator, generate_document_embeddings
from vector_store import RAGVectorStore, load_vector_store

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
app_state = {
    "vector_store": None,
    "data_loader": None,
    "embedding_generator": None,
    "ingestion_status": "not_started",
    "ingestion_progress": {}
}


# Pydantic models
class HealthResponse(BaseModel):
    status: str
    message: str
    components: Dict[str, str]


class IngestRequest(BaseModel):
    document_limit: Optional[int] = Field(default=None, description="Limit number of documents to ingest")
    chunk_size: int = Field(default=1000, description="Text chunk size for embedding", gt=0)
    batch_size: int = Field(default=5, description="Batch size for embedding generation", gt=0)
    overwrite: bool = Field(default=False, description="Overwrite existing index")
    use_ai_titles: bool = Field(default=False, description="Use AI (GPT-4o) to generate titles from passage content")

    @field_validator('document_limit')
    @classmethod
    def validate_document_limit(cls, v):
        if v is not None and v <= 0:
            raise ValueError('document_limit must be positive if specified')
        return v


class IngestResponse(BaseModel):
    status: str
    message: str
    documents_processed: int
    nodes_created: int
    embedding_stats: Dict[str, Any]


class QueryRequest(BaseModel):
    query: str = Field(..., description="Query to search for")
    top_k: int = Field(default=5, description="Number of results to return")
    similarity_threshold: float = Field(default=0.0, description="Minimum similarity threshold")


class QueryResult(BaseModel):
    rank: int
    text: str
    metadata: Dict[str, Any]
    similarity_score: Optional[float]
    node_id: str


class QueryResponse(BaseModel):
    query: str
    results: List[QueryResult]
    total_results: int


class RAGRequest(BaseModel):
    query: str = Field(..., description="Question to answer")
    top_k: int = Field(default=3, description="Number of documents to retrieve")


class RAGResponse(BaseModel):
    query: str
    answer: str
    sources: List[QueryResult]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    logger.info("Starting RAG application...")

    # Try to load existing vector store
    try:
        vector_store = load_vector_store()
        if vector_store:
            app_state["vector_store"] = vector_store
            app_state["ingestion_status"] = "completed"
            logger.info("Loaded existing vector store")
        else:
            logger.info("No existing vector store found")
    except Exception as e:
        logger.warning(f"Failed to load vector store: {e}")

    yield

    logger.info("Shutting down RAG application...")


app = FastAPI(
    title="Wikipedia RAG System",
    description="A Retrieval-Augmented Generation system for Wikipedia articles",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/chat")
async def chat_ui():
    """Serve the RAG chat interface."""
    chat_path = Path(__file__).parent / "static" / "chat.html"
    return FileResponse(chat_path, media_type="text/html")


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    components = {
        "vector_store": "loaded" if app_state["vector_store"] else "not_loaded",
        "ingestion_status": app_state["ingestion_status"],
        "data_loader": "available",
        "embedding_generator": "available"
    }

    return HealthResponse(
        status="healthy",
        message="Wikipedia RAG System is running",
        components=components
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_data(request: IngestRequest, background_tasks: BackgroundTasks):
    """Ingest Wikipedia data and create vector index."""

    if app_state["ingestion_status"] == "in_progress":
        raise HTTPException(status_code=409, detail="Ingestion already in progress")

    if app_state["vector_store"] and not request.overwrite:
        raise HTTPException(status_code=409, detail="Vector store already exists. Use overwrite=true to replace.")

    # Set status to in_progress BEFORE starting background task
    app_state["ingestion_status"] = "in_progress"
    app_state["ingestion_progress"] = {"step": "starting"}

    # Start background ingestion
    background_tasks.add_task(
        _ingest_data_background,
        request.document_limit,
        request.chunk_size,
        request.batch_size,
        request.overwrite,
        request.use_ai_titles
    )

    return IngestResponse(
        status="started",
        message="Data ingestion started in background",
        documents_processed=0,
        nodes_created=0,
        embedding_stats={}
    )


@app.get("/ingest/status")
async def get_ingestion_status():
    """Get current ingestion status."""
    return {
        "status": app_state["ingestion_status"],
        "progress": app_state.get("ingestion_progress", {})
    }


async def _ingest_data_background(document_limit: Optional[int],
                                 chunk_size: int,
                                 batch_size: int,
                                 overwrite: bool,
                                 use_ai_titles: bool):
    """Background task for data ingestion."""
    try:
        logger.info(f"Starting background ingestion: limit={document_limit}")

        # Load documents
        app_state["ingestion_progress"]["step"] = "loading_documents"
        if use_ai_titles:
            app_state["ingestion_progress"]["step"] = "loading_documents_with_ai_titles"
        documents = load_wikipedia_documents(limit=document_limit, use_ai_titles=use_ai_titles)

        app_state["ingestion_progress"]["documents_loaded"] = len(documents)
        logger.info(f"Loaded {len(documents)} documents")

        # Generate embeddings
        app_state["ingestion_progress"]["step"] = "generating_embeddings"
        nodes, stats = generate_document_embeddings(
            documents,
            chunk_size=chunk_size,
            batch_size=batch_size
        )

        app_state["ingestion_progress"]["nodes_created"] = len(nodes)
        logger.info(f"Generated embeddings for {len(nodes)} nodes")

        # Create vector store
        app_state["ingestion_progress"]["step"] = "creating_vector_store"
        vector_store = RAGVectorStore()
        vector_store.create_index_from_nodes(nodes)

        # Save vector store
        app_state["ingestion_progress"]["step"] = "saving_index"
        save_success = vector_store.save_index(overwrite=overwrite)

        if save_success:
            app_state["vector_store"] = vector_store
            app_state["ingestion_status"] = "completed"
            app_state["ingestion_progress"]["step"] = "completed"
            logger.info("Ingestion completed successfully")
        else:
            app_state["ingestion_status"] = "failed"
            app_state["ingestion_progress"]["step"] = "failed"
            logger.error("Failed to save vector index")

    except Exception as e:
        app_state["ingestion_status"] = "failed"
        app_state["ingestion_progress"]["error"] = str(e)
        logger.error(f"Ingestion failed: {e}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents using similarity search."""

    if not app_state["vector_store"]:
        raise HTTPException(status_code=404, detail="No vector store available. Run ingestion first.")

    try:
        results = app_state["vector_store"].query_similar_documents(
            query=request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold
        )

        query_results = [
            QueryResult(
                rank=result["rank"],
                text=result["text"],
                metadata=result["metadata"],
                similarity_score=result["similarity_score"],
                node_id=result["node_id"]
            )
            for result in results
        ]

        return QueryResponse(
            query=request.query,
            results=query_results,
            total_results=len(query_results)
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/rag", response_model=RAGResponse)
async def generate_answer(request: RAGRequest):
    """Generate answer using RAG (Retrieval-Augmented Generation)."""

    if not app_state["vector_store"]:
        raise HTTPException(status_code=404, detail="No vector store available. Run ingestion first.")

    try:
        # Get query engine
        query_engine = app_state["vector_store"].get_query_engine(
            similarity_top_k=request.top_k,
            response_mode="compact"
        )

        if not query_engine:
            raise HTTPException(status_code=500, detail="Failed to create query engine")

        # Generate answer
        response = query_engine.query(request.query)

        # Get source documents
        sources = []
        if hasattr(response, 'source_nodes') and response.source_nodes:
            for i, node in enumerate(response.source_nodes):
                sources.append(QueryResult(
                    rank=i + 1,
                    text=node.text,
                    metadata=node.metadata,
                    similarity_score=getattr(node, 'score', None),
                    node_id=node.node_id
                ))

        return RAGResponse(
            query=request.query,
            answer=str(response),
            sources=sources
        )

    except Exception as e:
        logger.error(f"RAG generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"RAG generation failed: {str(e)}")


@app.get("/info")
async def get_system_info():
    """Get system information."""
    info = {
        "ingestion_status": app_state["ingestion_status"],
        "vector_store_available": app_state["vector_store"] is not None
    }

    if app_state["vector_store"]:
        index_info = app_state["vector_store"].get_index_info()
        info.update(index_info)

    return info


@app.get("/knowledge")
async def get_knowledge_overview():
    """Get overview of knowledge base topics and content."""
    if not app_state["vector_store"]:
        raise HTTPException(status_code=404, detail="No vector store available. Run ingestion first.")

    try:
        # Sample some documents to understand the knowledge base
        sample_results = app_state["vector_store"].query_similar_documents(
            query="sample",  # Generic query to get diverse results
            top_k=50,  # Get more samples for analysis
            similarity_threshold=0.0  # Accept all results
        )

        if not sample_results:
            return {"topics": [], "total_documents": 0, "sample_titles": []}

        # Extract topics from metadata
        topics = {}
        titles = set()

        for result in sample_results:
            metadata = result.get("metadata", {})
            title = metadata.get("title", "Unknown")
            source = metadata.get("source", "Unknown")

            titles.add(title)

            # Group by source or try to infer topic categories
            if source not in topics:
                topics[source] = {
                    "count": 0,
                    "sample_titles": []
                }

            topics[source]["count"] += 1
            if len(topics[source]["sample_titles"]) < 5:
                topics[source]["sample_titles"].append(title)

        # Get total count from vector store info
        index_info = app_state["vector_store"].get_index_info()
        total_nodes = index_info.get("total_nodes", len(sample_results))

        return {
            "topics": topics,
            "total_documents": total_nodes,
            "unique_titles_sampled": len(titles),
            "sample_titles": sorted(list(titles))[:20]  # First 20 alphabetically
        }

    except Exception as e:
        logger.error(f"Failed to get knowledge overview: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze knowledge base: {str(e)}")