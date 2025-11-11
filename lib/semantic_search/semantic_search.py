from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import lib.utils.data_loader_utils as data_loader_utils
import lib.utils.constants as constants

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


class SemanticSearch:

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=constants.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"}
        )
        self.vector_db = None
        self.documents = []

    def build_embeddings(self):
        """
            Chunk all the documents and build embeddings out of it.
        """
        try:
            # Load the documents
            self.documents = data_loader_utils.load_about_me()

            # Chunk all the docs
            all_chunks = []
            for index, doc in enumerate(self.documents):
                chunks = self.semantic_chunk(doc, index)
                all_chunks.extend(chunks)

            print("All documents chunked successfully")
            for index, chunk in enumerate(all_chunks):
                print(f"{index + 1}. {chunk}")
                print("")

            # If all chunks are empty just return
            if not all_chunks:
                print(f"There are no chunks to save.")
                return

            # Embed and Save to ChromaDB
            print(
                f"Embedding and saving {len(all_chunks)} chunks to {constants.CHROMA_PATH}...")
            self.vector_db = Chroma.from_documents(
                documents=all_chunks,
                embedding=self.embeddings,
                persist_directory=constants.CHROMA_PATH,
                collection_name=constants.COLLECTION_NAME
            )

            print("Embedding process complete. Vector database built successfully.")

        except Exception as e:
            print(f"Unable to load the documents {e}")
            self.vector_db = None

    def semantic_chunk(self, document: Document, doc_idx: int, chunk_size: int = 500, overlap: int = 50) -> list[Document]:
        """
            Chunk the page_content of the input document into chunks. 
            Chunking is done in sentence by sentence format.
        """

        # Separator list for sentence awareness
        # TODO: - Find out how chunking works.
        custom_separators = [
            "\n\n",
            "\n",
            ". ",
            "! ",
            "? ",
            ", ",
            " ",
            ""
        ]

        # Initialize the TextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=custom_separators,
            is_separator_regex=False
        )

        chunked_documents = splitter.split_documents([document])
        for index, chunked_doc in enumerate(chunked_documents):
            chunked_doc.metadata["doc_idx"] = doc_idx
            chunked_doc.metadata["chunk_idx"] = index

        return chunked_documents
