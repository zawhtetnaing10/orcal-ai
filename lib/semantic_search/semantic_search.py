import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import lib.utils.data_loader_utils as data_loader_utils
import lib.utils.constants as constants

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class SemanticSearch:

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=constants.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"}
        )
        self.vector_db = None
        self.documents = []

    def semantic_search(self, query, limit=3) -> list[Document]:
        """
            Perform a semantic search according to query.
            Will prepare the embeddings first before searching.
        """
        if not os.path.exists(constants.CHROMA_PATH):
            print(
                f"Haven't built the embeddings yet. Please build the embeddings first before searching.")

        # Fetch double the amount of chunks since they'll have to be mapped back to documents
        chunks_with_scores = self.vector_db.similarity_search_with_score(
            query, limit * 2)

        # Map the chunks to a dict with doc_idx as key and score as value
        doc_idx_dict = {}
        for chunk, score in chunks_with_scores:
            doc_idx = int(chunk.metadata["doc_idx"])
            if doc_idx in doc_idx_dict:
                # If doc_id already in dict. Only overwrite if it has lower score (better similarity)
                previous_score = doc_idx_dict[doc_idx]
                if score < previous_score:
                    doc_idx_dict[doc_idx] = score
            else:
                # If not, just add the doc_idx
                doc_idx_dict[doc_idx] = score

        # Sort the dict items with score
        sorted_doc_idx_with_scores = sorted(
            doc_idx_dict.items(), key=lambda item: item[1])
        # Get the doc_idx
        sorted_doc_idx = [
            item[0] for item in sorted_doc_idx_with_scores
        ]

        # Map the doc_idx back to the original docs
        result = [
            self.documents[doc_idx] for doc_idx in sorted_doc_idx
        ]

        return result[:limit]

    def build_or_load_embeddings(self):
        """
            If embeddings are already built, just load them.
            If not, build the embeddings from scratch
        """

        # Load the documents first. To be sure
        if not self.documents:
            self.documents = data_loader_utils.load_about_me()

        if os.path.exists(constants.CHROMA_PATH):
            self.vector_db = Chroma(
                persist_directory=constants.CHROMA_PATH,
                embedding_function=self.embeddings,
                collection_name=constants.COLLECTION_NAME
            )
        else:
            self.build_embeddings()

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
