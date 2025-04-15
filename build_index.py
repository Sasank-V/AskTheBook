import fitz  # PyMuPDF
import os
import io
import re
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import math

IMAGE_DIR = "images"
FIGURE_DIR = "figures"


# ---------------------
# Utilities for image filtering
# ---------------------
def image_entropy(img):
    histogram = img.convert("L").histogram()
    total_pixels = sum(histogram)
    if total_pixels == 0:
        return 0
    entropy = -sum(
        (p / total_pixels) * math.log(p / total_pixels, 2) for p in histogram if p > 0
    )
    return entropy


def is_relevant_image(img, min_width=30, min_height=30, min_entropy=1.0, min_colors=3):
    """Filter out irrelevant images with adjustable parameters"""
    width, height = img.size

    if width < min_width or height < min_height:
        return False

    if image_entropy(img) < min_entropy:
        return False

    # Count unique colors (up to a reasonable limit for efficiency)
    colors = img.getcolors(maxcolors=1024)
    if colors is not None and len(colors) < min_colors:
        return False

    return True


# ---------------------
# Caption detector with improved pattern matching
# ---------------------
def find_figure_captions(page, page_num):
    """Find all figure captions with more comprehensive pattern matching"""
    # Extract all text from the page with their bounding boxes
    text_blocks = page.get_text("dict")["blocks"]
    caption_blocks = []

    # Regular expressions for different caption styles
    caption_patterns = [
        r"^(Figure|Fig\.?)\s+\d+(\.\d+)?",  # Figure 1.2, Fig. 3, etc.
        r"^(Table)\s+\d+(\.\d+)?",  # Table 1.2, etc.
        r"^(Diagram|Chart|Graph)\s+\d+(\.\d+)?",  # Diagram 1, Chart 2.3, etc.
    ]

    for block in text_blocks:
        if "lines" in block:
            for line in block["lines"]:
                if "spans" in line:
                    for span in line["spans"]:
                        text = span.get("text", "").strip()

                        # Check if this span matches any caption pattern
                        for pattern in caption_patterns:
                            if re.search(pattern, text, re.IGNORECASE):
                                # Found a caption, store its bounding box
                                caption_blocks.append(
                                    {
                                        "text": text,
                                        "bbox": [
                                            span["bbox"][0],
                                            span["bbox"][1],
                                            span["bbox"][2],
                                            span["bbox"][3],
                                        ],
                                    }
                                )
                                break

    print(f"Found {len(caption_blocks)} potential caption blocks on page {page_num+1}")
    return caption_blocks


# ---------------------
# Improved vector figure extraction using caption detection
# ---------------------
def extract_vector_figures(page, page_num):
    """
    Look for figure captions and extract the figure area above or around them
    with more sophisticated area detection
    """
    vector_figures = []
    caption_blocks = find_figure_captions(page, page_num)

    for idx, caption in enumerate(caption_blocks):
        caption_text = caption["text"]
        caption_rect = fitz.Rect(caption["bbox"])

        print(f"Processing caption on page {page_num+1}: '{caption_text}'")

        # Calculate a more appropriate region for the figure
        # We'll look for both above and slightly to the sides of the caption
        # Adaptive margin based on page dimensions
        page_height = page.rect.height
        page_width = page.rect.width

        # Vertical margin - look upward from caption for the figure
        # Use a percentage of page height (e.g., 25%) or minimum of 150 points
        vertical_margin = max(150, page_height * 0.5)

        # Horizontal margin - expand slightly beyond caption width
        # Use a percentage of page width (e.g., 5%) on each side
        horizontal_margin = page_width * 0.8

        # Create a rectangle that extends above the caption
        fig_x0 = max(0, caption_rect.x0 - horizontal_margin)
        fig_x1 = min(page_width, caption_rect.x1 + horizontal_margin)
        fig_y1 = caption_rect.y0  # Top of the caption is bottom of figure
        fig_y0 = max(0, fig_y1 - vertical_margin)  # Go upward from caption

        fig_rect = fitz.Rect(fig_x0, fig_y0, fig_x1, fig_y1)

        # Check if the figure area is reasonable
        if fig_rect.width < 50 or fig_rect.height < 50:
            print(f"  Skipping figure area that's too small: {fig_rect}")
            continue

        # Render the region as an image
        pix = page.get_pixmap(clip=fig_rect, dpi=300)  # Higher DPI for better quality
        os.makedirs(FIGURE_DIR, exist_ok=True)
        caption_id = re.sub(r"[^a-zA-Z0-9]", "_", caption_text[:20])
        vector_fig_path = f"{FIGURE_DIR}/fig_p{page_num+1}_{caption_id}.png"
        pix.save(vector_fig_path)
        vector_figures.append(
            {"path": vector_fig_path, "caption": caption_text, "page": page_num + 1}
        )

        print(f"  Saved figure to {vector_fig_path}")

    return vector_figures


# ---------------------
# Main PDF loader with improved extraction
# ---------------------
def load_pdf_with_images(pdf_path):
    print(f"Processing PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    documents = []
    all_figures = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        page_figures = []

        print(f"\nProcessing page {page_num+1}/{len(doc)}")

        # 1. Extract embedded images from the page
        image_count = 0
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"]

                try:
                    image = Image.open(io.BytesIO(image_bytes))

                    # Apply less restrictive filtering
                    if not is_relevant_image(
                        image, min_width=30, min_height=30, min_entropy=0.8
                    ):
                        print(
                            f"  Skipped embedded image {img_index}; size={image.size}, entropy={image_entropy(image):.2f}"
                        )
                        continue

                    os.makedirs(IMAGE_DIR, exist_ok=True)
                    img_path = f"{IMAGE_DIR}/page_{page_num+1}_{img_index}.{ext}"
                    with open(img_path, "wb") as img_out:
                        img_out.write(image_bytes)

                    page_figures.append(
                        {"path": img_path, "type": "embedded", "page": page_num + 1}
                    )
                    image_count += 1

                except Exception as e:
                    print(f"  Error processing embedded image: {e}")
            except Exception as e:
                print(f"  Error extracting image: {e}")

        print(f"  Extracted {image_count} embedded images")

        # 2. Extract vector-based figures using improved caption detection
        vector_figures = extract_vector_figures(page, page_num)
        for fig in vector_figures:
            page_figures.append(
                {
                    "path": fig["path"],
                    "type": "vector",
                    "caption": fig["caption"],
                    "page": page_num + 1,
                }
            )

        print(f"  Extracted {len(vector_figures)} vector figures")

        # 3. Add all figures to the overall collection
        all_figures.extend(page_figures)

        # 4. Create a Document with metadata containing page number and figure details
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "page": page_num + 1,
                    "figures": [fig["path"] for fig in page_figures],
                    "figure_details": page_figures,
                },
            )
        )

    print(f"\nTotal figures extracted: {len(all_figures)}")
    return documents, all_figures


# ---------------------
# Build and save the index
# ---------------------
def get_subject_from_filename(filename):
    return os.path.splitext(os.path.basename(filename))[0]  # e.g., OS from OS.pdf


def process_subject_pdf(pdf_path):
    subject = get_subject_from_filename(pdf_path)
    subject_index_path = f"index/{subject}"

    # Check if index already exists
    if os.path.exists(subject_index_path):
        print(f"✅ Skipping {subject} - already indexed.")
        return

    print(f"\n=== Processing subject: {subject} ===")

    docs, all_figures = load_pdf_with_images(pdf_path)

    os.makedirs("summaries", exist_ok=True)
    with open(f"summaries/{subject}_figure_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Total figures extracted: {len(all_figures)}\n\n")
        for i, fig in enumerate(all_figures):
            f.write(
                f"{i+1}. {fig['path']} (Type: {fig['type']}, Page: {fig['page']})\n"
            )
            if "caption" in fig:
                f.write(f"   Caption: {fig['caption']}\n")
            f.write("\n")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks for {subject}")

    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embedder)

    os.makedirs(subject_index_path, exist_ok=True)
    db.save_local(subject_index_path)
    print(f"✅ Saved FAISS index to {subject_index_path}")
    print(f"✅ Summary saved to summaries/{subject}_figure_summary.txt")


def main():
    pdf_folder = "data"
    os.makedirs("index", exist_ok=True)  # Ensure index folder exists

    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            process_subject_pdf(os.path.join(pdf_folder, filename))


if __name__ == "__main__":
    main()
