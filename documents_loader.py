
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path(__file__).parent / "data"


def load_documents():
    documents = []

    for file_path in DATA_DIR.glob("*.vtt"):
        text = file_path.read_text(encoding="utf-8")

        # Remove the VTT header and timestamps
        text = re.sub(r"(?m)^WEBVTT\s*$", "", text)
        text = re.sub(
            r"(?m)^\d{2}:\d{2}:\d{2}\.\d+\s+-->\s+.*$",
            "",
            text,
        )

        # Remove blank lines and join transcript segments
        text = "\n".join(
            line.strip()
            for line in text.splitlines()
            if line.strip()
        )

        documents.append(
            Document(
                page_content=text,
                metadata={"source": file_path.name},
            )
        )

    if not documents:
        raise ValueError(f"No VTT files found in {DATA_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    return splitter.split_documents(documents)
