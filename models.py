import streamlit as st
from google import genai
from google.genai.errors import ClientError, ServerError
from google.genai import types
from openai import OpenAI
import time
import threading
from multiprocessing.pool import ThreadPool
from typing import List, Literal
from pydantic import BaseModel


gpt_key = st.secrets["openai"]["api_key"]
gemini_keys = [s['api_key'] for s in st.secrets["gemini"]]
gemini_clients = [genai.Client(api_key=key) for key in gemini_keys]
oai_client = OpenAI(api_key=gpt_key)

GENERATE_CATEGORIES_PROMPT = """
You are a smart data assistant.

The user will provide a sample of several rows from a larger Excel table. Each row represents a real-world data entry.

Your task is to analyze these rows and suggest relevant and general **categories** in **Hebrew** that could be used to classify similar data in the full table.

Instructions:
- Output the categories in **Hebrew only** (no English).
- The categories should be **general**, not overly specific.
- The categories must be **relevant** to the data entries provided.
- The final category must always be **"אחר"** (meaning "other").

Respond only with the list of suggested categories in Hebrew.
"""

CLASSIFY_TEXT_PROMPT = (
    "You are a classification assistant. Your task is to classify the given text and choose the most appropriate categories from the following list:\n\n"
    "{classes}\n\n"
    "Given a text, return explanations and category names list\n\n"
    "Rules:\n"
    "1. You may select one or more categories.\n"
    "2. Before each category you choose, **explain** why you chose it.\n"
    "3. If you select ANY category other than 'אחר', then 'אחר' MUST NOT appear in your answer.\n"
    "4. If there are NO other matching categories at all, then return ONLY 'אחר'.\n"
    "5. Do not choose the same category twice.\n\n"
)

CLASSIFY_TEXT_PROMPT_ONE = (
    "You are a classification assistant. Your task is to assign the most suitable category to the given feedback text based on the following list of categories:\n\n"
    "{classes}\n\n"
    "Before you choose the right category, **explain** why you chose it.\n\n"
    "Guidelines:\n"
    "1. Choose **exactly one** category from the list above.\n"
    "2. Match based on **intent and topic**, not just specific words. Look for the **underlying meaning** of the feedback.\n"
    "3. If the feedback clearly relates to any item in the list — even if indirectly — choose the most related category.\n\n"
)

def first_classification_ai(df, columns_to_classify):
    client = gemini_clients[0]
    if not columns_to_classify:
        raise ValueError("No columns selected for classification.")

    sample_text = "\n".join(df[columns_to_classify].astype(str).iloc[:, 0].dropna().head(3000).tolist())  # Take first column's sample
    response = ""
    response = client.models.generate_content(model="gemini-2.0-flash",
                                                config=types.GenerateContentConfig(
                                                    responseMimeType="application/json",
                                                    responseSchema=list[str],
                                                    system_instruction=GENERATE_CATEGORIES_PROMPT,
                                                ),
                                                contents=sample_text,
                                                )
    classes = response.parsed
    return classes

def get_classes(texts, classes, retries=2, delay=5, client=gemini_clients[0],update_progress=None):
    if not classes:
        raise ValueError("No classes provided for classification.")
    if not texts:
        return None
    class Output(BaseModel):
        explanation: str
        category: Literal[tuple(classes)]
    for _ in range(retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                config=types.GenerateContentConfig(
                    system_instruction=CLASSIFY_TEXT_PROMPT.format(classes=classes),
                    # system_instruction=CLASSIFY_TEXT_PROMPT_ONE.format(classes=classes),
                    responseMimeType="application/json",
                    responseSchema=list[Output],
                    # responseSchema=Output,
                    temperature=0.0,
                ),
                contents="\n".join(texts),
            )
            #remove double quotes from the output
            if update_progress:
                update_progress()
            return response.parsed
        except ClientError as e:
            if e.code == 429:
                print("Rate Limit Error:", e)
                time.sleep(delay)
            else:
                raise e
        except ServerError as e:
            if e.code == 503:
                print("Resource Unavailable Error:", e)
                time.sleep(delay)
            else:
                raise e
    if update_progress:
        update_progress()
    return None

client_id = -1

def classify_data(data, classes, num_thread=4, retries=2, delay=5, clients=gemini_clients, update_progress=None):
    if not classes:
        raise ValueError("No classes provided for classification.")
    
    lock = threading.Lock()
    
    def init_worker(clients):
        global client_id
        with lock:
            client_id = (client_id + 1) % len(clients)
    def worker(*args, **kwargs):
        global client_id
        client = clients[client_id]
        return get_classes(*args, client=client, **kwargs, update_progress=update_progress)

    with ThreadPool(num_thread, initializer=init_worker, initargs=(clients,)) as pool:
        results = pool.starmap(worker, [(val, classes, retries, delay) for val in data])
        
    return [list(set(r.category for r in res)) if res else None for res in results]
    # return [r.category if r else None for r in results]

def first_classification_ai_gpt(df, columns_to_classify):
    if not columns_to_classify:
        raise ValueError("No columns selected for classification.")
    sample_text = "\n".join(df[columns_to_classify].astype(str).iloc[:, 0].dropna().head(3000).tolist())  # Take first column's sample
    response = oai_client.responses.create(
        model="gpt-4.1-mini",
        instructions=GENERATE_CATEGORIES_PROMPT,
        input=sample_text,
        temperature=0,
    )
    classes = response.output_text.split("\n")
    return classes

def get_classes_gpt(text, classes, retries=2, delay=5):
    class Output(BaseModel):
        explanation: str
        category: Literal[tuple(classes)]
    class OutputList(BaseModel):
        outputs: List[Output]
    
    for _ in range(retries + 1):
        try:
            response = oai_client.responses.parse(
                model="gpt-4.1-mini",
                instructions=CLASSIFY_TEXT_PROMPT.format(classes=classes),
                input=text,
                temperature=0,
                text_format=OutputList,
                # text_format=Output,
            )
            return response.output_parsed
        except Exception as e:
            print("error", e)
            time.sleep(delay)
    return None

def classify_data_gpt(data, classes, num_thread=8, retries=2, delay=5):
    if not classes:
        raise ValueError("No classes provided for classification.")

    with ThreadPool(num_thread) as pool:
        results = pool.starmap(get_classes_gpt, [(val, classes, retries, delay) for val in data])

    return [list(set([r.category for r in res.outputs])) for res in results]
    # return [r.category if r else None for r in results]

