from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class SemanticSearch:
    def semantic_chunk(self, document: Document, doc_idx: int, chunk_size: int = 500, overlap: int = 50) -> list[Document]:
        """
            Chunk the page_content of the input document into chunks. 
            Chunking is done in sentence by sentence format.
        """

        # TODO: - Find out how chunking works.
        # Separator list for sentence awareness
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
