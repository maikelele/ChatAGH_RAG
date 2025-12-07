from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass
class RetrievedContext:
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
