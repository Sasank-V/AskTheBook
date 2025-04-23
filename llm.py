import os
import ollama
from constants import subject_descriptions, subjects
from vector_store import query_vector_store, get_page, get_page_image_paths
from constants import pdfs_path, model, model_manim, model_r1, temp_path
import json
from pydantic import BaseModel
from typing import List
import re
import tempfile
import subprocess
from langchain_core.output_parsers.string import StrOutputParser


def get_models_list():
    models_response = ollama.list()
    print(models_response.models)
    models = []
    for model in models_response.models:
        models.append(model.model)
    return models


def generate_response(model_name, prompt, images_path=[]):
    images = []
    for path in images_path:
        with open(path, "rb") as img:
            images.append(img.read())
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt, "images": images}],
    )
    return response.message.content


def classify_subjects(model, query):
    prompt = f"""
    You are an expert assistant that classifies academic queries into relevant subjects based on their descriptions.
    Subjects and their descriptions:
    {chr(10).join([f"- {s}: {d}" for s, d in subject_descriptions.items()])}

    Given the question:
    '{query}'

    List the subjects (comma-separated) that are most relevant to the question.
    Only choose from: {', '.join(subject_descriptions.keys())}
    """
    response = generate_response(model, prompt=prompt)
    subjects = [sub.strip() for sub in response.split(",")]
    return subjects


def generate_similar_queries(model, query, num_queries=3):
    prompt = f"""
        You are an Expert AI assistant that helps rewrite questions in different ways to explore various angles, terminologies, and styles of inquiry. 

        Given the question:
        '{query}'

        Generate {num_queries} distinct rephrasings of this question, ensuring that:
        - The core intent of the original question remains intact.
        - Each rephrasing explores a slightly different perspective, wording, or approach.
        - Use academic and subject-specific vocabulary where applicable.
        - Make the rephrasings natural and useful for a student or researcher looking for information.

        Provide each variation on a new line, without numbering or bullet points.
        """
    response = generate_response(model, prompt=prompt)
    queries = [q.strip() for q in response.strip().split("\n") if q.strip()]
    queries.insert(0, query)
    return list(set(queries))[:5]


def search_book_indexes(queries, books):
    results = {}
    for query in queries:
        response = query_vector_store(query=query, booknames=books)
        for sub in response.keys():
            if sub not in results.keys():
                results[sub] = response[sub]
            else:
                results[sub] |= response[sub]
    return results


def summarise_pages(model, book, pages):
    full_summary = ""
    for i in range(0, len(pages), 3):
        full_text = ""
        for page_idx in pages[i : i + 3]:
            text = get_page(book_name=book, page_num=page_idx)
            full_text += text + "\n"
        prompt = f"""
        Carefully read the content from the following textbook page(s):
        {full_text}
        
        Extract and summarize all relevant and important information, including definitions, key concepts, steps in processes, examples, and any data or diagrams provided. Do not skip any critical detail. Ensure that the summary retains the full meaning and educational value of the original content while presenting it in a more concise and easy-to-read format suitable for revision.
        """

        response = generate_response(model_name=model, prompt=prompt)
        full_summary += f"\nSummary for {book} pages {pages[i:i+3]}:\n{response}\n"

    return full_summary


class StichResponse(BaseModel):
    answer: str
    relevant_pages: List[int]


def stitch_response(model, query, books: dict[str, list[int]]):
    results = {}
    for book in books.keys():
        # print("Book:", book)
        pages = books[book]
        full_text = {}

        for page_idx in pages:
            full_text[page_idx] = get_page(book_name=book, page_num=page_idx)

        prompt = f"""
You are a powerful LLM assistant that answers user queries by referring to textbook pages and your own knowledge.
Each page is represented by a **real page number** and contains text and optional images.

You are provided with:
- A **user query**
- A collection of **textbook pages**, each labeled by their **actual page number** and containing associated content

Your job is to:
1. Read and understand the textbook pages deeply
2. Generate a clear, structured, and accurate answer to the user query using the textbook pages and your own knowledge

### User Query:
{query}

### Textbook Pages:
{{
{chr(10).join([f'"Page {k}": "{v}"' for k, v in full_text.items()])}
}}

"""

        # print(prompt)
        res = generate_response(model_name=model, prompt=prompt)
        think_match = re.search(r"<think>(.*?)</think>", res, re.DOTALL)
        think_text = think_match.group(1).strip() if think_match else ""
        remaining_text = re.sub(r"<think>.*?</think>", "", res, flags=re.DOTALL).strip()
        results[book] = {"think": think_text, "answer": remaining_text}
    return results


def get_animation_prompt(model, query, image_paths):
    prompt = f"""
    
🧠 Prompt: Generate a Detailed Manim Scene Plan Based on User Query and Supporting Images

You are a creative and intelligent educational animation planner. Your job is to create a visually engaging and instructive scene idea for the Manim Python library that teaches a concept clearly to a learner.

You are given:
1. A **user question or topic** to explain.
2. One or more **relevant images** (like diagrams, graphs, figures, handwritten notes, etc.) that provide visual hints about the concept.

Your task is to:
✅ Analyze the user query and the provided images together.
✅ Design a full **animation plan** using visual storytelling — not code.
✅ Focus only on the images given and the query (ignore unrelated visuals).
✅ Use your scene planning to make the concept **clear, visual, and memorable**.
Ask the LLM to return the python code as python raw string

---

⚠️ IMPORTANT:
- DO NOT write actual Python or Manim code.
- Describe the **visual structure**, sequence, and animation style in detail.
- Mention how Manim elements like `Text`, `MathTex`, `Square`, `NumberPlane`, `VGroup`, etc. would be used.

---

📥 User Query:
```
{query}
```

🖼️ Relevant Images Provided: {{len(image_paths)}} image(s)
- Only consider the content in these images. Do NOT assume or invent extra visuals.

---

📤 Output Structure:

🎬 Scene Title: <Descriptive name for the scene>

🎨 Concept Overview:
<A short paragraph about what this animation aims to explain and why it's important>

🧩 Step-by-Step Breakdown:
1. <Start with a basic shape, equation, or idea — describe the object and animation technique (e.g., FadeIn, Write, Transform)>
2. <Next, explain how it evolves — add lines, arrows, equations, comparisons>
3. <Integrate visual cues from the images — describe how an image part is animated or visualized>
4. <Include use of layout strategies like NumberPlane for graphs or VGroup for alignment>
5. <Continue until the concept is fully explained with engaging visual flow>

🎥 Final Frame:
<Describe the final visual state of the scene. What should the learner walk away remembering?>

💡 Notes for Coders:
<Give tips for the dev team or LLM that will convert this to Manim code — e.g., “Use Transform for concept transitions”, “Use ValueTracker to animate dynamic quantities”, or “Use MathTex for equations from the image.”>
The final ClassName of the Scene should AnimationScene , eg: class AnimationScene(Scene): 

---

🧠 Think visually. Be pedagogically smart. Help the next LLM generate stunning Manim scenes to teach complex concepts with elegance.
    """
    return generate_response(model_name=model, prompt=prompt, images_path=image_paths)


def get_animation(query, image_paths):
    prompt = get_animation_prompt(model=model, query=query, image_paths=image_paths)
    print("Prompt:", prompt)

    raw_output = generate_response(model_name=model_manim, prompt=prompt)
    parser = StrOutputParser()
    code = parser.parse(raw_output)

    match = re.search(r"```(?:python)?\n(.*?)```", code, re.DOTALL)
    if match:
        code = match.group(1).strip()

    return code.strip()


def run_manim_code(code):
    temp_dir = os.path.join(os.getcwd(), "temp")
    os.makedirs(temp_dir, exist_ok=True)

    code_file_path = os.path.join(temp_dir, "generated_scene.py")
    with open(code_file_path, "w", newline="\n") as f:
        f.write(code)

    result = subprocess.run(
        [
            "manim",
            code_file_path,
            "-pql",
            "-o",
            "output",
        ],
        cwd=temp_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        error_message = result.stderr.decode()
        raise Exception(f"Manim rendering failed: {error_message}")

    rendered_path = os.path.join(
        temp_dir, "media", "videos", "generated_scene", "480p15", "output.mp4"
    )
    if not os.path.exists(rendered_path):
        raise Exception("Output video not found!")

    return rendered_path


def to_raw(string):
    return rf"{string}"


if __name__ == "__main__":
    query = "What is a CNN ? "
    # subjects = classify_subjects(model=model, query=query)
    # print("Subjects: ", subjects)
    # queries = generate_similar_queries(model=model, query=query, num_queries=5)
    # print("Queries: ", queries)
    # result = search_book_indexes(books=subjects, queries=queries)
    # print("Relevant: ", result)
    # image_paths = []
    # for book in result.keys():
    #     pages = result[book]
    #     for page in pages:
    #         image_paths += get_page_image_paths(book_name=book, page_num=page)
    # print("Image Paths: ", image_paths)
    image_paths = [
        "images\\AI\\page_961_img_0_0.png",
        "images\\AI\\page_962_img_0_0.png",
        "images\\AI\\page_772_img_0_0.png",
        "images\\AI\\page_965_img_0_0.png",
        "images\\AI\\page_746_img_0_0.png",
        "images\\AI\\page_746_img_0_1.png",
        "images\\AI\\page_746_img_1_0.png",
        "images\\AI\\page_746_img_1_1.png",
        "images\\AI\\page_746_img_2_0.png",
        "images\\AI\\page_746_img_2_1.png",
        "images\\AI\\page_746_img_3_0.png",
        "images\\AI\\page_746_img_3_1.png",
        "images\\AI\\page_746_img_4_0.png",
        "images\\AI\\page_746_img_5_0.png",
        "images\\AI\\page_747_img_0_0.png",
        "images\\AI\\page_747_img_0_1.png",
        "images\\AI\\page_747_img_1_0.png",
        "images\\AI\\page_747_img_1_1.png",
        "images\\AI\\page_752_img_0_0.png",
        "images\\AI\\page_752_img_0_1.png",
        "images\\AI\\page_752_img_1_0.png",
        "images\\AI\\page_752_img_1_1.png",
        "images\\AI\\page_980_img_0_0.png",
        "images\\AI\\page_980_img_0_1.png",
        "images\\AI\\page_980_img_1_0.png",
        "images\\AI\\page_980_img_1_1.png",
        "images\\AI\\page_29_img_0_0.png",
    ]
    code = get_animation(query=query, image_paths=image_paths)
    print("Code:", code)
    print(run_manim_code(code=code))
