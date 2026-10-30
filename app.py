
import streamlit as st
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Paper Knowledge Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Paper Knowledge Assistant")
st.write(
    "Ask questions about paper policies and documents. "
    "Answers are based on the documents stored in the knowledge base."
)


# ---------------- DOCUMENT FOLDER ----------------

DOCUMENTS_FOLDER = Path(__file__).parent / "documents"


# ---------------- LOAD DOCUMENTS ----------------

@st.cache_resource
def build_knowledge_base():

    documents = []

    if not DOCUMENTS_FOLDER.exists():
        return None, "The documents folder was not found."

    files = [
        file for file in DOCUMENTS_FOLDER.iterdir()
        if file.is_file() and file.suffix.lower() in [".pdf", ".txt"]
    ]

    if len(files) < 5:
        return None, (
            f"Found only {len(files)} documents. "
            "Please add at least 5 PDF or TXT files."
        )

    for file in files:
        try:
            if file.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file))
                loaded_docs = loader.load()

            else:
                loader = TextLoader(
                    str(file),
                    encoding="utf-8",
                    autodetect_encoding=True
                )
                loaded_docs = loader.load()

            for doc in loaded_docs:
                doc.metadata["source"] = file.name

            documents.extend(loaded_docs)

        except Exception as error:
            st.warning(f"Could not read {file.name}: {error}")

    if not documents:
        return None, "No readable document content was found."

    # Split documents into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        return None, "No text chunks could be created."

    # Create embeddings and FAISS vector database
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    database = FAISS.from_documents(chunks, embeddings)

    return database, (
        f"Loaded {len(files)} files, "
        f"{len(documents)} pages/documents, "
        f"and created {len(chunks)} searchable chunks."
    )


# ---------------- BUILD KNOWLEDGE BASE ----------------

with st.spinner("Reading documents and building the knowledge base..."):
    database, status_message = build_knowledge_base()

if database is None:
    st.error(status_message)
    st.info(
        "Place at least five PDF or TXT files inside the documents folder, "
        "then restart the app."
    )
    st.stop()

st.success(status_message)


# ---------------- QUESTION INPUT ----------------

question = st.text_input(
    "Ask your question",
    placeholder="Example: What is observatory?"
)

search_button = st.button("Get Answer", type="primary")


# ---------------- RETRIEVAL AND ANSWERING ----------------

if search_button:

    if not question.strip():
        st.warning("Please enter a question.")

    else:
        with st.spinner("Searching documents..."):

            # Retrieve the most relevant chunks
            results = database.similarity_search_with_score(
                question,
                k=4
            )

            # Lower FAISS L2 distance generally means a closer match.
            # This is a configurable safeguard, not a universal threshold.
            distance_threshold = 1.2

            relevant_results = [
                (doc, score)
                for doc, score in results
                if score <= distance_threshold
                and doc.page_content.strip()
            ]

        if not relevant_results:
            st.warning(
                "I could not find sufficiently relevant information "
                "in the available documents to answer this question."
            )

        else:
            context_parts = []

            for index, (doc, score) in enumerate(
                relevant_results, start=1
            ):
                source = doc.metadata.get("source", "Unknown file")
                page = doc.metadata.get("page")

                if page is not None:
                    source_info = f"{source}, page {page + 1}"
                else:
                    source_info = source

                context_parts.append(
                    f"[Source {index}: {source_info}]\n"
                    f"{doc.page_content}"
                )

            context = "\n\n".join(context_parts)

            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    """
You are a Paper Knowledge Assistant.

Rules:
1. Answer only using the supplied document excerpts.
2. Do not invent facts, rules, dates, amounts, or policies.
3. If the excerpts do not contain enough evidence, say:
   "The available documents do not provide enough information
   to answer this question."
4. If sources conflict, clearly explain the contradiction and
   identify both source documents. Do not arbitrarily choose one.
5. Be concise and answer the exact question.
6. Mention the source filename when supporting a factual answer.

Document excerpts:
{context}
"""
                ),
                ("human", "{question}")
            ])

            llm = OllamaLLM(
                model="gemma2:2b",
                temperature=0
            )

            chain = prompt | llm | StrOutputParser()

            with st.spinner("Generating answer with Gemma..."):
                try:
                    answer = chain.invoke({
                        "context": context,
                        "question": question
                    })

                    st.subheader("Answer")
                    st.write(answer)

                    st.subheader("Supporting Sources")

                    for index, (doc, score) in enumerate(
                        relevant_results, start=1
                    ):
                        source = doc.metadata.get(
                            "source", "Unknown file"
                        )
                        page = doc.metadata.get("page")

                        if page is not None:
                            location = f"Page {page + 1}"
                        else:
                            location = "Page not available"

                        with st.expander(
                            f"{index}. {source} - {location}"
                        ):
                            st.write(doc.page_content)
                            st.caption(
                                f"FAISS L2 distance: {float(score):.4f}"
                            )

                except Exception as error:
                    st.error(
                        "Could not generate the answer. "
                        "Check that Ollama is running and Gemma is installed."
                    )
                    st.code(str(error))


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.header("About this application")
    st.write("**Technique:** Retrieval-Augmented Generation (RAG)")
    st.write("**Interface:** Streamlit")
    st.write("**Vector database:** FAISS")
    st.write("**Embedding model:** all-MiniLM-L6-v2")
    st.write("**Language model:** Gemma 2:2B via Ollama")
    st.write("**Retrieval count:** 4 chunks")
    st.caption(
        "Answers depend on document quality and retrieval accuracy. "
        "Always verify important policies against the original documents."
    )