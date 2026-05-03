"""
Resource Retrieval Agent Module


This module defines an agent that retrieves and processes academic resources
using a RAG (Retrieval-Augmented Generation) workflow.


The agent fetches content from educational websites focused on:
- Python error handling and debugging
- Instructional and teaching strategies
- Learning and tutoring best practices


This agent informs other tutoring agents with research-based academic resources
to ensure outputs are grounded in educational research and best practices.


The agent uses:
- OpenAI embeddings for semantic understanding
- FAISS for efficient vector similarity search
- LangChain for workflow orchestration
"""


import os
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from abc import ABC


# Set USER_AGENT to avoid warnings from requests library
os.environ['USER_AGENT'] = 'ResourceRetrievalAgent/1.0 (Educational Research Tool)'


from dotenv import load_dotenv
from openai import OpenAI


from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Load environment variables from .env file
load_dotenv()




@dataclass
class ResourceCategory:
   """
   Represents a category of educational resources.
  
   Attributes:
       name (str): Category name (e.g., "Python Error Handling")
       urls (List[str]): List of URLs containing resources for this category
       description (str): Description of what this category covers
   """
   name: str
   urls: List[str]
   description: str




class ResourceRetrievalAgent(ABC):
   """
   Agent responsible for retrieving and processing academic resources via RAG.
  
   This agent fetches educational content from curated websites, indexes them
   using FAISS vector database, and retrieves relevant resources based on queries.
   Other tutoring agents can use this agent to access research-based materials
   that inform their instructional content.
  
   Features:
   - Multi-source document retrieval
   - Semantic similarity search via embeddings
   - Citation tracking and provenance
   - Cached vector index for efficiency
   - Research-grounded resource curation
  
   Attributes:
       name (str): The name/role of the agent
       openai_client (OpenAI): The OpenAI client instance
       embeddings_model (str): The embedding model to use
       llm (ChatOpenAI): The language model for processing
       vector_store (FAISS): The indexed vector database
       resource_categories (Dict): Organized resource categories
   """
  
   # Default configuration
   EMBEDDINGS_MODEL = "text-embedding-3-small"
   LLM_MODEL = "gpt-4o-mini"
   CHUNK_SIZE = 300
   CHUNK_OVERLAP = 50
   RETRIEVAL_K = 6
  
   # Resource categories with curated URLs
   RESOURCE_CATEGORIES = {
       "python_errors": ResourceCategory(
           name="Python Error Handling & Debugging",
           urls=[
               "https://docs.python.org/3/tutorial/errors.html",
               "https://thepythoncodingbook.com/errors-and-bugs/"
           ],
           description="Official Python documentation and resources on error handling, exceptions, and debugging techniques"
       ),
       "instructional_strategies": ResourceCategory(
           name="Instructional & Teaching Strategies",
           urls=[
               "https://crlt.umich.edu/resources/teaching-strategies",
               "https://www.discoveryeducation.com/blog/teaching-and-learning/instructional-strategies/"
           ],
           description="Evidence-based teaching strategies, instructional design principles, and pedagogical best practices"
       )
   }
  
   def __init__(self, name: str = "Resource Retrieval Specialist"):
       """
       Initialize the Resource Retrieval Agent.
      
       Args:
           name (str): The name or role identifier for this agent
                      (default: "Resource Retrieval Specialist")
      
       Raises:
           KeyError: If OPENAI_API_KEY environment variable is not set
           Exception: If resource loading fails
       """
       self.name = name
      
       # Initialize OpenAI client with API key from environment
       try:
           api_key = os.getenv("OPENAI_API_KEY")
           if not api_key:
               raise KeyError("OPENAI_API_KEY environment variable not set")
           self.openai_client = OpenAI(api_key=api_key)
       except KeyError as e:
           raise KeyError(f"Failed to initialize OpenAI client: {e}")
      
       # Initialize embedding model and LLM
       self.embeddings_model = self.EMBEDDINGS_MODEL
       self.embeddings = OpenAIEmbeddings(model=self.embeddings_model)
       self.llm = ChatOpenAI(model=self.LLM_MODEL, temperature=0.2)
      
       # Initialize vector store
       self.vector_store: Optional[FAISS] = None
       self.documents: List[Document] = []
       self.load_status: Dict[str, bool] = {}
      
       # Load all resources from configured URLs
       self._load_all_resources()
  
   def _load_all_resources(self) -> None:
       """
       Load and index all resources from configured URLs.
      
       This method loads content from all URLs in RESOURCE_CATEGORIES,
       cleans the text, chunks it, and builds a FAISS vector index.
       """
       print(f"[{self.name}] Initializing resource retrieval system...")
       all_documents = []
      
       for category_key, category in self.RESOURCE_CATEGORIES.items():
           print(f"\n  Loading resources from category: {category.name}")
           for url in category.urls:
               try:
                   print(f"    • Fetching: {url}")
                   docs = self._load_and_process_url(url, category.name)
                   all_documents.extend(docs)
                   self.load_status[url] = True
                   print(f"      ✓ Loaded {len(docs)} chunks")
               except Exception as e:
                   self.load_status[url] = False
                   print(f"      ✗ Error: {str(e)}")
      
       if all_documents:
           self.documents = all_documents
           self._build_vector_store(all_documents)
           print(f"\n✓ Resource retrieval system initialized with {len(all_documents)} document chunks")
       else:
           raise RuntimeError("Failed to load any resources from configured URLs")
  
   def _load_and_process_url(self, url: str, category: str) -> List[Document]:
       """
       Load content from a URL, clean it, and split into chunks.
      
       Args:
           url (str): The URL to fetch
           category (str): The resource category for metadata
          
       Returns:
           List[Document]: List of chunked documents with metadata
          
       Raises:
           Exception: If content cannot be loaded or processed
       """
       # Load content from URL
       headers = {
           "User-Agent": "ResourceRetrievalAgent/1.0 (Educational Research Tool)"
       }
       loader = WebBaseLoader(url, header_template=headers)
       docs = loader.load()
      
       if not docs:
           raise ValueError(f"No content retrieved from {url}")
      
       # Clean and normalize content
       raw_content = docs[0].page_content
       clean_content = re.sub(r'\s+', ' ', raw_content).strip()
      
       if not clean_content:
           raise ValueError(f"Content at {url} is empty after cleaning")
      
       # Create a document with metadata
       cleaned_doc = Document(
           page_content=clean_content,
           metadata={
               "source": url,
               "category": category,
               "loaded_at": datetime.now().isoformat()
           }
       )
      
       # Split into chunks
       text_splitter = RecursiveCharacterTextSplitter(
           chunk_size=self.CHUNK_SIZE,
           chunk_overlap=self.CHUNK_OVERLAP
       )
       documents = text_splitter.split_documents([cleaned_doc])
      
       # Add chunk indices for citation tracking
       for i, doc in enumerate(documents):
           doc.metadata["chunk_index"] = i
      
       return documents
  
   def _build_vector_store(self, documents: List[Document]) -> None:
       """
       Build FAISS vector store from documents.
      
       Args:
           documents (List[Document]): Documents to index
       """
       self.vector_store = FAISS.from_documents(documents, self.embeddings)
  
   def retrieve_resources(
       self,
       query: str,
       k: int = None,
       category_filter: str = None
   ) -> List[Dict]:
       """
       Retrieve relevant resources based on a semantic query.
      
       Args:
           query (str): The search query
           k (int): Number of results to retrieve (default: RETRIEVAL_K)
           category_filter (str): Optional category to filter results
          
       Returns:
           List[Dict]: List of retrieved resources with metadata
       """
       if not self.vector_store:
           raise RuntimeError("Vector store not initialized")
      
       k = k or self.RETRIEVAL_K
      
       # Retrieve documents
       retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
       retrieved_docs = retriever.invoke(query)
      
       # Apply category filter if specified
       if category_filter:
           retrieved_docs = [
               doc for doc in retrieved_docs
               if doc.metadata.get("category") == category_filter
           ]
      
       # Format results
       resources = []
       for doc in retrieved_docs:
           resources.append({
               "content": doc.page_content,
               "source": doc.metadata.get("source", "unknown"),
               "category": doc.metadata.get("category", "uncategorized"),
               "chunk_index": doc.metadata.get("chunk_index", 0),
               "loaded_at": doc.metadata.get("loaded_at", "unknown")
           })
      
       return resources
  
   def answer_with_context(self, query: str, context_resources: List[Dict] = None) -> Dict:
       """
       Answer a query using retrieved resources as context.
      
       Args:
           query (str): The user's question
           context_resources (List[Dict]): Optional pre-retrieved resources
          
       Returns:
           Dict: Answer with citations and retrieved resources
       """
       # Retrieve resources if not provided
       if not context_resources:
           context_resources = self.retrieve_resources(query)
      
       if not context_resources:
           return {
               "query": query,
               "answer": "No relevant resources found for this query.",
               "resources": [],
               "citations": []
           }
      
       # Format context from resources
       context_text = "\n\n".join([r["content"] for r in context_resources])
      
       # Create prompt template
       prompt = ChatPromptTemplate.from_template("""Answer the following question based on the provided context from academic and educational resources:


**Context:**
{context}


**Question:** {query}


Provide a clear, research-based answer drawing from the context provided.""")
      
       # Build and invoke chain
       chain = (
           {
               "context": lambda x: context_text,
               "query": lambda x: x["query"]
           }
           | prompt
           | self.llm
           | StrOutputParser()
       )
      
       answer = chain.invoke({"query": query})
      
       return {
           "query": query,
           "answer": answer,
           "resources": context_resources,
           "citations": self._generate_citations(context_resources)
       }
  
   @staticmethod
   def _generate_citations(resources: List[Dict]) -> List[str]:
       """Generate citation strings from resources."""
       citations = []
       for i, resource in enumerate(resources, 1):
           source = resource["source"]
           category = resource["category"]
           chunk = resource["chunk_index"]
           snippet = resource["content"][:100].replace("\n", " ")
           citation = f"{i}. [{category}] {source} (chunk {chunk})\n   Snippet: {snippet}..."
           citations.append(citation)
       return citations
  
   def get_category_resources(self, category_name: str) -> List[Dict]:
       """
       Get all resources from a specific category.
      
       Args:
           category_name (str): The category name
          
       Returns:
           List[Dict]: All resources in the category
       """
       if not self.vector_store:
           raise RuntimeError("Vector store not initialized")
      
       # Return all documents from the category
       resources = []
       for doc in self.documents:
           if doc.metadata.get("category") == category_name:
               resources.append({
                   "content": doc.page_content,
                   "source": doc.metadata.get("source"),
                   "category": doc.metadata.get("category"),
                   "chunk_index": doc.metadata.get("chunk_index")
               })
       return resources
  
   def generate_tutor_guidance(self, topic: str, student_level: str) -> Dict:
       """
       Generate tutoring guidance for a specific topic using retrieved resources.
      
       This method is designed to support other tutoring agents by providing
       research-based instructional guidance.
      
       Args:
           topic (str): The tutoring topic
           student_level (str): The student's proficiency level
          
       Returns:
           Dict: Comprehensive tutoring guidance with resources and strategies
       """
       # Retrieve relevant resources
       query = f"How to teach {topic} to {student_level} students"
       resources = self.retrieve_resources(query, k=8)
      
       # Generate guidance using LLM
       prompt = ChatPromptTemplate.from_template("""Based on these educational resources and best practices,
provide comprehensive tutoring guidance:


**Resources:**
{resources}


**Topic:** {topic}
**Student Level:** {student_level}


Provide:
1. Key learning objectives
2. Most effective instructional strategies
3. Common misconceptions to address
4. Recommended practice activities
5. Assessment methods""")
      
       resources_text = "\n\n".join([r["content"][:200] for r in resources])
      
       chain = (
           {
               "resources": lambda x: resources_text,
               "topic": lambda x: x["topic"],
               "student_level": lambda x: x["student_level"]
           }
           | prompt
           | self.llm
           | StrOutputParser()
       )
      
       guidance = chain.invoke({
           "topic": topic,
           "student_level": student_level
       })
      
       return {
           "topic": topic,
           "student_level": student_level,
           "guidance": guidance,
           "source_resources": resources,
           "citations": self._generate_citations(resources)
       }
  
   def get_resource_summary(self) -> Dict:
       """
       Get a summary of all loaded resources.
      
       Returns:
           Dict: Summary statistics and load status
       """
       summary = {
           "agent": self.name,
           "total_chunks": len(self.documents),
           "embeddings_model": self.embeddings_model,
           "llm_model": self.LLM_MODEL,
           "categories": {},
           "load_status": self.load_status,
           "timestamp": datetime.now().isoformat()
       }
      
       # Count chunks by category
       for category in self.RESOURCE_CATEGORIES.values():
           category_docs = [
               d for d in self.documents
               if d.metadata.get("category") == category.name
           ]
           summary["categories"][category.name] = {
               "url_count": len(category.urls),
               "chunk_count": len(category_docs),
               "description": category.description
           }
      
       return summary
  
   def execute(self, query: str, action: str = "answer") -> Dict:
       """
       Execute the resource retrieval agent.
      
       Args:
           query (str): The query or topic
           action (str): The action to perform:
                        - "answer": Answer a question with resources
                        - "retrieve": Retrieve relevant resources
                        - "guidance": Generate tutoring guidance
                        - "summary": Get resource summary
          
       Returns:
           Dict: Results based on the action
       """
       if action == "answer":
           return self.answer_with_context(query)
       elif action == "retrieve":
           return {
               "query": query,
               "resources": self.retrieve_resources(query),
               "resource_count": len(self.retrieve_resources(query))
           }
       elif action == "guidance":
           # Extract topic and level from query if available
           # Format: "topic | student_level"
           parts = query.split("|")
           topic = parts[0].strip() if parts else query
           student_level = parts[1].strip() if len(parts) > 1 else "Intermediate"
           return self.generate_tutor_guidance(topic, student_level)
       elif action == "summary":
           return self.get_resource_summary()
       else:
           raise ValueError(f"Unknown action: {action}")



