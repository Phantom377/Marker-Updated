import re

def split_by_headings(markdown):
    pattern = r'(#{1,6} .+)'
    parts = re.split(pattern, markdown)

    chunks = []
    current = ""

    for part in parts:
        if part.startswith("#"):
            if current:
                chunks.append(current.strip())
            current = part
        else:
            current += "\n" + part

    if current:
        chunks.append(current.strip())

    return chunks

def merge_tables(chunks):
    merged = []
    buffer = ""

    for chunk in chunks:
        if "|" in chunk:  # crude table detection
            buffer += "\n" + chunk
        else:
            if buffer:
                merged.append(buffer.strip())
                buffer = ""
            merged.append(chunk)

    if buffer:
        merged.append(buffer.strip())

    return merged

def split_large_chunks(chunks, max_chars=1000):
    final_chunks = []

    for chunk in chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
        else:
            for i in range(0, len(chunk), max_chars):
                final_chunks.append(chunk[i:i+max_chars])

    return final_chunks

def chunk_markdown(markdown):
    chunks = split_by_headings(markdown)
    chunks = merge_tables(chunks)
    chunks = split_large_chunks(chunks)
    return chunks