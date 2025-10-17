from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Chunk:
    """
    data container for a text chunk

    :param section: section name
    :param chunk_id: index of the chunk within the section
    :param text: chunk contents
    """
    section: str
    chunk_id: int
    text: str


class TextSplitter:
    """
    split text into overlapping word-based chunks
    """

    def __init__(self, chunk_size: int = 250, overlap: int = 50):
        """
        initialize text splitter with chunking parameters

        :param chunk_size: desired number of words per chunk
        :param overlap: number of words to overlap between chunks
        """
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _split_words(self, text: str) -> List[str]:
        """
        split one string into overlapping word windows

        :param text: input text
        :return: list of chunk strings
        """
        if not text:
            return []

        words = text.split()
        chunks: List[str] = []
        i = 0
        
        while i < len(words):
            window = words[i : i + self.chunk_size]
            chunk = " ".join(window)
            chunks.append(chunk)
            
            if i + self.chunk_size >= len(words):
                break
            
            i += self.chunk_size - self.overlap
        
        return chunks

    def split_section(self, section_name: str, section_text: str) -> List[Chunk]:
        """
        split a single section into chunk objects

        :param section_name: section name
        :param section_text: full text of the section
        :return: list of chunk objects
        """
        parts = self._split_words(section_text)
        return [
            Chunk(section=section_name, chunk_id=i, text=p) 
            for i, p in enumerate(parts)
        ]

    def split_sections(self, sections: Dict[str, str]) -> List[Chunk]:
        """
        split multiple sections into chunk objects

        :param sections: mapping of section name to text
        :return: list of chunk objects across all sections
        """
        all_chunks: List[Chunk] = []
        for name, text in sections.items():
            section_chunks = self.split_section(name, text)
            all_chunks.extend(section_chunks)
        return all_chunks

    def summary(self, chunks: List[Chunk]) -> str:
        """
        return a short summary string of chunk statistics

        :param chunks: list of chunk objects
        :return: human-readable summary
        """
        if not chunks:
            return "no chunks generated"
        
        lengths = [len(c.text.split()) for c in chunks]
        avg_len = sum(lengths) / len(lengths)
        
        return f"{len(chunks)} chunks, avg length {avg_len:.1f} words"
