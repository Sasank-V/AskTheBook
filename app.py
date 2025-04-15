import os
import streamlit as st
from rag_qa import load_groq_llm, load_qa_chain
import random

# Set up Streamlit page configuration
st.set_page_config(page_title="Ask the Book", layout="wide")
st.title("📄 Ask the Book")

# List of subjects and corresponding index directories
SUBJECTS = ["AI", "CAO-1", "CVLA-1", "CVLA-2", "DBMS", "OS"]
INDEX_DIR = "./index/"


# Function to ask LLM which subjects are relevant to the query
def classify_subjects(query, llm):
    subject_descriptions = {
        "AI": "Artificial Intelligence - includes search algorithms, machine learning, reasoning, and problem solving.",
        "CAO-1": "Computer Architecture and Organization - focuses on CPU, memory, assembly, and computer system design.",
        "CVLA-1": "Linear Algebra - involves matrices, vector spaces, eigenvalues, and linear transformations.",
        "CVLA-2": "Complex Variables - deals with complex numbers, analytic functions, residues, and contour integration.",
        "DBMS": "Database Management Systems - covers relational models, SQL, normalization, and transactions.",
        "OS": "Operating Systems - includes process management, memory, scheduling, file systems, and concurrency.",
    }

    prompt = f"""
You are an assistant that classifies academic queries into relevant subjects based on their descriptions.

Subjects and their descriptions:
{chr(10).join([f"- {s}: {d}" for s, d in subject_descriptions.items()])}

Given the question:
'{query}'

List the subjects (comma-separated) that are most relevant to the question.
Only choose from: {', '.join(subject_descriptions.keys())}
"""

    try:
        response = llm.invoke(prompt)
        subjects = [
            s.strip()
            for s in response.content.split(",")
            if s.strip() in subject_descriptions
        ]
        return subjects
    except Exception as e:
        st.error(f"Error classifying subjects: {e}")
        return []


# Function to generate similar queries using LLM
def generate_similar_queries_llm(query, llm, num_queries=3):
    try:
        prompt = f"""
        You are an AI assistant that helps rewrite questions in different ways to explore various angles, terminologies, and styles of inquiry. 

        Given the question:
        '{query}'

        Generate {num_queries} distinct rephrasings of this question, ensuring that:
        - The core intent of the original question remains intact.
        - Each rephrasing explores a slightly different perspective, wording, or approach.
        - Use academic and subject-specific vocabulary where applicable.
        - Make the rephrasings natural and useful for a student or researcher looking for information.

        Provide each variation on a new line, without numbering or bullet points.
        """
        response = llm.invoke(prompt)
        queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
        queries.insert(0, query)
        return list(set(queries))[:5]
    except Exception as e:
        st.error(f"Error generating similar queries: {e}")
        return [query]


# Function to process queries across relevant subjects
def process_queries_across_subjects(queries, subjects):
    all_answers = []
    all_sources = []

    for subject in subjects:
        qa_chain = load_qa_chain(subject)
        for query in queries:
            try:
                response = qa_chain({"query": query})
                all_answers.append(response["result"])
                for doc in response["source_documents"]:
                    if not any(
                        d.metadata.get("page") == doc.metadata.get("page")
                        and d.page_content == doc.page_content
                        for d in all_sources
                    ):
                        all_sources.append(doc)
            except Exception as e:
                st.error(f"Error processing '{query}' in {subject}: {e}")
    return all_answers, all_sources


# Summarize answers using LLM
def summarize_answers(answers, llm, query):
    if not answers:
        return "No relevant information found."
    try:
        combined = "\n".join([f"- {a}" for a in answers])
        prompt = (
            f"Summarize the following answers for the question '{query}':\n{combined}"
        )
        summary_response = llm.invoke(prompt)
        return summary_response.content.strip()
    except Exception as e:
        st.error(f"Error summarizing: {e}")
        return "Could not summarize the answers."


# Streamlit UI
query = st.text_input("Enter your question")

if query:
    groq_llm = load_groq_llm()

    with st.spinner("Classifying query to relevant subjects..."):
        relevant_subjects = classify_subjects(query, groq_llm)
        st.success(f"📚 Relevant subjects: {', '.join(relevant_subjects)}")

    if not relevant_subjects:
        st.warning("Could not detect relevant subjects.")
    else:
        similar_queries = generate_similar_queries_llm(query, groq_llm)

        st.subheader("🧠 Similar Queries Generated")
        for i, q in enumerate(similar_queries):
            st.markdown(f"- {q}")

        with st.spinner("Fetching answers..."):
            answers, sources = process_queries_across_subjects(
                similar_queries, relevant_subjects
            )
            final_answer = summarize_answers(answers, groq_llm, query)

        st.subheader("📌 Final Answer")
        st.markdown(final_answer)

        st.subheader("📖 Source Pages & Figures")
        if not sources:
            st.info("No relevant source documents found.")
        else:
            seen_pages = set()
            for doc in sources:
                page = doc.metadata.get("page")
                if page not in seen_pages:
                    seen_pages.add(page)
                    st.markdown(f"---\n**Source: Page {page}**")
                    st.markdown(f"**Snippet:** {doc.page_content[:500]}...")

                    figures = doc.metadata.get("figure_details", [])
                    if figures:
                        st.markdown("**Figures on this page:**")
                        for fig in figures:
                            path = fig["path"]
                            if os.path.exists(path):
                                st.image(
                                    path,
                                    caption=f"{fig['type']} | Page {fig['page']}\n{fig.get('caption', 'No caption')}",
                                    use_container_width=True,
                                )
                            else:
                                st.warning(f"Image not found: {path}")
                    else:
                        st.info("No figures found.")
