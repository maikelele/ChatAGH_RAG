from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass
class RetrievedContext:
    """
    Container for retrieved and augmented context associated with a single source URL.

    Attributes:
        source_url (str):
            The URL from which the primary retrieved chunks originate.

        chunks (list[Document]):
            The list of document chunks directly retrieved during the initial
            similarity search for this URL.

        related_chunks (dict[str, list[Document]]):
            A mapping of related URLs to their corresponding chunks, obtained
            through graph-based context augmentation. These represent additional
            context semantically linked to the primary source page.
    """

    source_url: str
    chunks: list[Document]
    related_chunks: dict[str, list[Document]]

    @property
    def text(self) -> str:
        chunks_text = "\n".join(chunk.page_content for chunk in self.chunks)
        related_chunks_text = "\n".join(
            f"{key}: {' | '.join(doc.page_content for doc in value)}"
            for key, value in self.related_chunks.items()
        )
        text = (
            f"Source URL: {self.source_url}\n"
            f"Retrieved chunks: {chunks_text}\n"
            f"Related context: {related_chunks_text}"
        )
        return text

    def __str__(self) -> str:
        return self.text
