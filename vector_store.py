import fitz
import os
import re

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from constants import images_path, captions_path, index_path, pdfs_path

os.makedirs(images_path, exist_ok=True)
os.makedirs(captions_path, exist_ok=True)
os.makedirs(index_path, exist_ok=True)
search_words = ["Figure", "Table", "Chart"]
pattern = r"(Figure|Table|Chart)\s\d+\.\d+"

min_width = 100
min_height = 100

model = SentenceTransformer("all-MiniLM-L6-v2")


def get_page_image_paths(book_name, page_num):
    paths = []
    folder_path = os.path.join(images_path, book_name)
    pattern = re.compile(rf"page_{page_num}_img.*\.png", re.IGNORECASE)
    if not os.path.isdir(folder_path):
        print(f"Folder '{folder_path}' not found.")
        return paths

    for filename in os.listdir(folder_path):
        if pattern.search(filename):
            paths.append(os.path.join(folder_path, filename))

    return paths


def get_page(book_name, page_num):
    book_path = os.path.join(pdfs_path, f"{book_name}.pdf")
    doc = fitz.open(book_path)
    text = doc[page_num].get_text()
    return text


def get_pdf_paths(folder_path):
    pdf_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(root, file))

    return pdf_paths


def get_embedding(text):
    return model.encode(text)


def store_text_embeddings(book_name, pdf_path):
    doc = fitz.open(pdf_path)
    embeddings = []
    embedding_dim = 384
    index = faiss.IndexFlatL2(embedding_dim)

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = page.get_text()
        embedding = get_embedding(text)
        embeddings.append(embedding)

    embedding_array = np.array(embeddings).astype("float32")
    index.add(embedding_array)

    index_file_path = os.path.join(index_path, f"{book_name}.index")
    faiss.write_index(index, index_file_path)
    print(f"Index for {book_name} saved at {index_file_path}")


def store_images_and_captions(book_name, pdf_path):
    doc = fitz.open(pdf_path)

    image_folder_path = os.path.join(images_path, book_name)
    captions_folder_path = os.path.join(captions_path, book_name)

    os.makedirs(image_folder_path, exist_ok=True)
    os.makedirs(captions_folder_path, exist_ok=True)

    for page_idx in range(len(doc)):
        page = doc[page_idx]

        text_blocks = page.get_text("blocks")
        text_blocks.sort(key=lambda b: (b[1], b[0]))
        text = page.get_text()
        matches = re.finditer(pattern, text)
        for match_idx, match in enumerate(matches):
            matched_text = match.group(0)
            rects = page.search_for(matched_text)
            if rects:
                for i, r in enumerate(rects):
                    padded_rect = fitz.Rect(
                        r.x0 - 100, r.y0 - 400, r.x1 + 400, r.y1 + 50
                    )
                    pix = page.get_pixmap(clip=padded_rect, dpi=300)
                    img_filename = f"page_{page_idx+1}_img_{match_idx}_{i}.png"
                    img_file_path = os.path.join(image_folder_path, img_filename)
                    pix.save(img_file_path)

                    following_line = ""
                    for idx, block in enumerate(text_blocks):
                        if matched_text in block[4]:
                            following_line = block[4]
                            break
                    if following_line:
                        caption_filename = img_filename.replace(".png", ".txt")
                        captions_filepath = os.path.join(
                            captions_folder_path, caption_filename
                        )
                        with open(
                            captions_filepath,
                            "w",
                            encoding="utf-8",
                        ) as f:
                            f.write(following_line)
    print(f"Images and captions for {book_name} Stored")


def index_all_pdfs():
    pdfpaths = get_pdf_paths(pdfs_path)
    for path in pdfpaths:
        bookname = os.path.splitext(os.path.basename(path))[0]
        print("Book:", bookname)
        store_images_and_captions(book_name=bookname, pdf_path=path)
        store_text_embeddings(book_name=bookname, pdf_path=path)


# index_all_pdfs()


def query_vector_store(query, booknames):
    query_embedding = get_embedding(query)
    query_embedding = np.array([query_embedding]).astype("float32")
    k = 5
    results = {}
    for book in booknames:
        path = os.path.join(index_path, f"{book}.index")
        index = faiss.read_index(path)
        D, I = index.search(query_embedding, k)
        results[book] = set(I.tolist()[0])
    return results
