import os
import streamlit as st
from llm import (
    generate_response,
    get_models_list,
    classify_subjects,
    generate_similar_queries,
    search_book_indexes,
    stitch_response,
)
from vector_store import get_page_image_paths
import random
from constants import model_r1, model, model_manim

st.set_page_config(page_title="Ask the Book", layout="wide")
st.title("📄 Ask the Book")
query = st.text_input("Enter your question")

if query:
    with st.spinner("Classifying query to relevant subjects..."):
        relevant_subjects = classify_subjects(model=model, query=query)
        st.success(f"📚 Relevant subjects: {', '.join(relevant_subjects)}")
    if not relevant_subjects:
        st.warning("Could not detect relevant subjects.")
    else:
        similar_queries = generate_similar_queries(model=model, query=query)
        st.subheader("🧠 Similar Queries Generated")
        for i, q in enumerate(similar_queries):
            st.markdown(f"- {q}")

        with st.spinner("Fetching answers..."):
            result = search_book_indexes(
                books=relevant_subjects, queries=similar_queries
            )
            final_answer = stitch_response(books=result, model=model_r1, query=query)

        st.subheader("📌 Final Answer")
        for book in final_answer.keys():
            st.markdown(f"## {book} Perspective")
            st.markdown("### Thought Process:")
            st.markdown(f"{final_answer[book]['think']}")

            st.markdown("### Final Answer:")
            st.markdown(
                f"{final_answer[book]['answer']}",
            )

        st.subheader("📖 Figures")
        for book in result.keys():
            st.markdown(f"- {book}")
            image_paths = []
            for page in result[book]:
                image_paths.append(get_page_image_paths(book_name=book, page_num=page))
            flattened_image_paths = [img for sublist in image_paths for img in sublist]
            st.image(flattened_image_paths)
