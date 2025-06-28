

import streamlit as st
import os
import re
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
from typing import List, Dict, Any
import openai
from pinecone import Pinecone, ServerlessSpec
import json
from openai import OpenAI
import time
import matplotlib.pyplot as plt
import base64
# from diagram_generator import load_json_from_s3, extract_resources, draw_diagram
# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
print("OPENAI API key:", openai.api_key)

# Check and set Pinecone API key
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY is not set in the environment variables.")
pc = Pinecone(api_key=pinecone_api_key)
print("Pinecone API key:", pinecone_api_key)

# Function to create the index with retry logic
def create_index_with_retry(index_name, dimension, metric, spec, retries=3, delay=5):
    for attempt in range(retries):
        try:
            if index_name not in pc.list_indexes().names():
                pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric=metric,
                    spec=spec
                )
            return True
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                st.error(f"Failed to create index after {retries} attempts: {e}")
                return False

index_name = "swap-redis-gcp"
dimension = 1536
metric = 'euclidean'
spec = ServerlessSpec(cloud='aws', region='us-east-1')

# Attempt to create the index
index_created = create_index_with_retry(index_name, dimension, metric, spec)

MODEL = "gpt-4o-mini"

class PenTestVAPTAssistant:
    def __init__(self, index_name, embeddings_model="text-embedding-3-small", llm_model=MODEL):
        self.openai = openai
        self.pinecone = pc
        self.index = self.pinecone.Index(index_name)
        self.embeddings_model = embeddings_model
        self.llm_model = llm_model
        self.client = OpenAI()

    def generate_embedding(self, text):
        try:
            response = self.client.embeddings.create(input=text, model=self.embeddings_model)
            return response.data[0].embedding
        except Exception as e:
            st.error(f"Error generating embedding: {e}")
            return None
    
    def search_index(self, query, top_k=6):
        embedding = self.generate_embedding(query)
        if embedding is None:
            return None
        try:
            query_result = self.index.query(
                vector=embedding,
                top_k=top_k,
                include_values=False,
                include_metadata=True
            )
            return query_result
        except Exception as e:
            st.error(f"Error querying index: {e}")
            return None

    def retrieve_documents(self, query_result, max_docs=3):
        documents = []
        if not query_result or 'matches' not in query_result:
            return documents
        for result in query_result['matches'][:max_docs]:
            try:
                document_text = result['metadata']['content']
                documents.append(document_text)
            except KeyError:
                st.error(f"Document ID '{result['id']}' not found in metadata.")
        return documents

    def generate_report(self, query, documents):
        prompt = f"Question: {query}\n\nRelevant Documents:\n"
        for doc in documents:
            prompt += f"- {doc}\n"
        prompt += "\nProvide a detailed answer to the question above based on the relevant documents. Include references in the format 'Reference: [source information]'." 

        role_description = (
    "= Your Role =\n"
    "Your primary role is to act as a technical assistant capable of answering detailed queries about Google Cloud Platform (GCP) infrastructure. "
    "You do this by analyzing and retrieving accurate information from a structured JSON dataset that represents the current state of GCP resources (buckets, instances, firewalls, etc.) indexed in Pinecone.\n\n"

    "= Your Knowledge Base =\n"
    "- Your knowledge base consists exclusively of a structured JSON dump of GCP infrastructure resources ingested into Pinecone.\n"
    "- You do not use general knowledge or assumptions about GCP. You rely strictly on the actual ingested dataset.\n"
    "- You understand the JSON structure and the fields of each GCP resource type (such as `encryption`, `lifecycle`, `status`, `name`, etc.).\n\n"

    "= Your Job =\n"
    "Upon receiving a user query related to GCP resources (e.g., number of buckets, instances without encryption, existence of lifecycle rules), "
    "you will:\n"
    "- Search and extract relevant data from the Pinecone-indexed JSON documents.\n"
    "- Interpret the structure to deliver precise, fact-based answers.\n"
    "- Present the results clearly with supporting logic and references to the relevant data points.\n"
    "- If the query is unclear or missing specifics (like project ID, resource type, etc.), ask the user for clarification.\n\n"

    "= How to Work with User Queries =\n"
    "1. Clarify ambiguous or incomplete questions by requesting specific details from the user.\n"
    "2. Parse the question to determine the intent (e.g., count, configuration, compliance check).\n"
    "3. Search the ingested Pinecone documents to extract matching JSON data.\n"
    "4. Analyze and summarize the relevant portions with supporting explanations.\n"
    "5. Present the results in structured, clear, and concise language. Use tables or bullet points when helpful.\n"
    "6. Do not hallucinate or assume missing data — only respond based on what is found in the indexed dataset.\n"
    "7. If data is missing or incomplete, state that clearly.\n\n"

    "= Example Queries You Should Be Able to Handle =\n"
    "- \"How many buckets are configured in our GCP project?\"\n"
    "- \"List all VM instances without encryption enabled.\"\n"
    "- \"Do any S3 (Cloud Storage) buckets have lifecycle rules defined?\"\n"
    "- \"Which instances are using public IPs?\"\n"
    "- \"Show firewall rules that allow 0.0.0.0/0 ingress.\"\n\n"

    "= Outputs =\n"
    "Your output should include:\n"
    "- A direct answer to the question based on JSON data from Pinecone.\n"
    "- References to the matching JSON fields and values (e.g., `bucket.lifecycle`, `instance.encryption.enabled`, etc.).\n"
    "- Explanation of how the answer was derived.\n"
    "- Structured formatting when applicable (e.g., tables or bullet points).\n"
    "- Clarification questions if the query lacks context or specifics.\n"
)

        
        messages = [
            {"role": "system", "content": role_description},
            {"role": "user", "content": prompt}
        ]
        
        completion_params = {
            "model": self.llm_model,
            "messages": messages
        }
        
        try:
            response = self.client.chat.completions.create(**completion_params)
            report = response.choices[0].message.content.strip()
            
            references = self.extract_references(report)
            
            return report, references
        except Exception as e:
            st.error(f"An error occurred while generating the report: {e}")
            return None, []

    def extract_references(self, report):
        lines = report.split("\n")
        references = [line for line in lines if line.startswith("Reference:")]
        return references

    def query(self, query):
        query_result = self.search_index(query)
        if query_result is None:
            return None, []
        documents = self.retrieve_documents(query_result)
        if not documents:
            st.warning("No relevant documents found.")
            return None, []
        report, references = self.generate_report(query, documents)
        return report, references


# Streamlit app layout
st.set_page_config(page_title="NCRB-GPT", layout="wide")

# Custom CSS for stylingststre
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
        }
        .main-title {
            font-size: 2.5rem;
            color: #000000;
            text-align: center;
            margin-bottom: 25px;
        }
        .description {
            font-size: 1.2rem;
            color: #333;
            text-align: center;
            margin-bottom: 50px;
        }
        .sidebar .sidebar-content {
            background-color: #F8F8FF;
        }
        .dataframe {
            margin: 20px;
            border-collapse: collapse;
        }
        .dataframe th {
            background-color: #4B0082;
            color: white;
        }
        .dataframe td, .dataframe th {
            padding: 10px;
            border: 1px solid #ddd;
        }
        .expander-header {
            font-size: 1.5rem;
            color: #4B0082;
        }
        .expander-content {
            font-size: 1.1rem;
            color: #555;
        }
        .expander-content p {
            margin: 10px 0;
        }
        input[type="text"] {
            autocomplete: off;
        }
    </style>
""", unsafe_allow_html=True)


# Title and description
st.markdown("<h1 class='main-title'>GCP-GPT</h1>", unsafe_allow_html=True)
st.markdown("<p class='description'>This dashboard allows you to ask questions and get answers from the model. Your question history will be displayed on the left-hand side.</p>", unsafe_allow_html=True)


# User interaction - Real-time chatbot
with st.form(key='question_form'):
    user_question = st.text_input("Enter your question here:", autocomplete='off')
    submit_button = st.form_submit_button(label='Ask')

# Initialize session state for storing history
if 'history' not in st.session_state:
    st.session_state.history = []

# Create the PenTestVAPTAssistant instance
assistant = PenTestVAPTAssistant(index_name=index_name)

# Placeholder for the answer
answer_placeholder = st.empty()
references_placeholder = st.empty()
chart_placeholder = st.empty()

if submit_button:
    if user_question.strip() != "":
        # Clear the previous answer and chart
        answer_placeholder.empty()
        references_placeholder.empty()
        chart_placeholder.empty()

        report, references = assistant.query(user_question)
        if report:
            st.session_state.history.append({
                "question": user_question,
                "answer": report,
                "references": references
            })
            
            # Display the new answer
            answer_placeholder.markdown(f"**Answer:**")
            lines = report.split("\n")
            print(lines)
            for line in lines:
                if "[\text" in line:  # Check if the line contains LaTeX
                    st.latex(line)
                else:
                    st.markdown(line)
            # Display references if any
            if references:
                references_placeholder.markdown("**References:**")
                for ref in references:
                    references_placeholder.markdown(f"- {ref}")

            # Display a sample chart
            
        else:
            st.write("No response generated.")
    else:
        st.write("Please enter a question.")

# Display question history only if there are questions in the history
if st.session_state.history:
    st.sidebar.write("### Question History")
    for i, entry in enumerate(st.session_state.history):
        if st.sidebar.button(entry['question'], key=f"history_{i}"):
            user_question = entry['question']
            answer_placeholder.empty()
            answer_placeholder.markdown(f"**Answer:** {entry['answer']}")
            references_placeholder.empty()
            if entry['references']:
                references_placeholder.markdown("**References:**")
                for ref in entry['references']:
                    references_placeholder.markdown(f"- {ref}")
            chart_placeholder.empty()

            