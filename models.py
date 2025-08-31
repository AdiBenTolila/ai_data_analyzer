import streamlit as st
from google import genai
from google.genai.errors import ClientError, ServerError
from google.genai import types
from openai import OpenAI, RateLimitError, InternalServerError, ContentFilterFinishReasonError
import time
import threading
from multiprocessing.pool import ThreadPool
from typing import List, Literal
from pydantic import BaseModel
from pydantic import Field
from openai import AzureOpenAI
from openai import BadRequestError


gpt_key = st.secrets["openai"]["api_key"]
gemini_keys = [s['api_key'] for s in st.secrets["gemini"]]
gemini_clients = [genai.Client(api_key=key) for key in gemini_keys]
oai_client = OpenAI(api_key=gpt_key)
az_client = AzureOpenAI(
    api_version=st.secrets["azure"]["api_version"],
    azure_endpoint=st.secrets["azure"]["endpoint"],
    api_key=st.secrets["azure"]["api_key"],
)

GENERATE_CATEGORIES_PROMPT = """
You are a smart data assistant.
 
The user will provide a sample of several rows from a larger Excel table. Each row represents a real-world data entry.
 
Your task is to analyze these rows and suggest relevant and general **categories** in **Hebrew** that could be used to classify similar data in the full table.
 
Instructions:
- Output the categories in Hebrew only (no English).
- Do not include feelings, moods, or general impressions.
- Categories should be clear and focused, but not micro-detailed.
- Do not use any broad or vague terms, or anything similar to: customer service, participant experience, community experience, atmosphere, or general production/organization.
- Categories must be relevant to the provided data entries.
- Use only letters, numbers, spaces, commas, and parentheses.
- The last category must always be "אחר" (meaning "other").
- Do not exceed 10 categories in total.
- There must be no duplication or near-duplication within the 10 categories (e.g., "Direction and information for participants" and "Signage and direction" are not allowed together).
Important: If any forbidden category or a similar broad term is included, the output will be considered incorrect.
 
"""

CLASSIFY_TEXT_PROMPT = (
    "You are a classification assistant. Your task is to classify the given text and choose the most appropriate categories from the following list:\n\n"
    "```{classes}```\n\n"
    "Also, choose one tag that describes best the opinion in the text, it should be one of the following tags:\n"
    "- מרוצה: the text expresses satisfaction with the service or product.\n"
    "- לא מרוצה: the text expresses dissatisfaction with the service or product.\n"
    "- ניטרלי: the text is neutral and does not express a clear opinion.\n"
    "Given a text, return explanations and category names list\n\n"
    "Rules:\n"
    "1. You have to select one or more categories.\n"
    "2. Before selecting categories and tag, **explain** why you chose them.\n"
    "3. If you select ANY category other than 'אחר', then 'אחר' MUST NOT appear in your answer.\n"
    "4. If there are NO matching categories at all, then return ONLY 'אחר'.\n"
    "5. Do not choose the same category twice.\n"
    "6. There must be at least 1 category and not more than 2 categories for the text.\n\n"
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
    if texts is None or len(texts) == 0 or all(str(text).strip() == "" for text in texts.values()):
        return []
    class Output(BaseModel):
        explanation: str
        category: Literal[tuple(classes)]
    if len(texts) > 1:
        prompt_content = "\n".join([f"{key}: {text}" for key,text in texts.items()])
    else:
        prompt_content = str(next(iter(texts.values())))
    for _ in range(retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                config=types.GenerateContentConfig(
                    system_instruction=CLASSIFY_TEXT_PROMPT.format(classes=classes),
                    responseMimeType="application/json",
                    responseSchema=list[Output],
                    temperature=0.0,
                ),
                contents=prompt_content,
            )
            #remove double quotes from the output
            if update_progress:
                update_progress()
            outputs = response.parsed
            filtered_outputs = [r for r in outputs if r.category != 'אחר']
            if filtered_outputs:
                return filtered_outputs
            else:
                return outputs
        
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
    return []

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
                model="gpt-4.1-nano",
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

def first_classification_ai_azure(df, columns_to_classify):
    class CategoryList(BaseModel):
        explanation: str
        categories: list[str]

    if not columns_to_classify:
        raise ValueError("No columns selected for classification.")
    sample_size = min(200, len(df))
    sample_text = "\n".join([str(record) for record in df[columns_to_classify].astype(str).dropna().sample(n=sample_size, random_state=42).to_dict(orient='records')])
    response = az_client.beta.chat.completions.parse(
        model=st.secrets["azure"]["categories_model_name"],
        messages=[
            {"role": "system", "content": GENERATE_CATEGORIES_PROMPT},
            {"role": "user", "content": sample_text}
        ],
        response_format=CategoryList,
        temperature=0.4,
    )
    classes = response.choices[0].message.parsed.categories
    return classes

def get_classes_azure(texts, classes, retries=2, delay=5, client=None, update_progress=None):
    if not classes:
        raise ValueError("No classes provided for classification.")
    if texts is None or len(texts) == 0 or all(str(text).strip() == "" for text in texts.values()):
        return {
            "explanation": "",
            "categories": [],
            "tag": 'ניטרלי'
        }

    available_tags = ('מרוצה', 'לא מרוצה','ניטרלי')

    class Output(BaseModel):
        explanation: str
        categories: List[Literal[tuple(classes)]]
        tag: Literal[available_tags]

    if len(texts) > 1:
        prompt_content = "\n".join([f"{key}: {text}" for key, text in texts.items()])
    else:
        prompt_content = str(next(iter(texts.values())))

    messages_model=[
                    {"role": "system", "content": CLASSIFY_TEXT_PROMPT.format(classes=classes)},
                    {"role": "user", "content": prompt_content}
                ]
    for attempt in range(retries + 1):
        try:
            response = az_client.beta.chat.completions.parse(
                model=st.secrets["azure"]["model_name"],
                messages=messages_model,
                temperature=0,
                response_format=Output,
            )

            if update_progress:
                update_progress()

            msg = response.choices[0].message
            output = msg.parsed
           
            if 'אחר' in output.categories and len(output.categories) > 1:
                output.categories = [c for c in output.categories if c != 'אחר']

            if not output.categories: 
                raw_text = getattr(msg, "content", "")
                if isinstance(raw_text, list):
                    raw_text = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in raw_text)

                messages_model.append({"role": "assistant", "content": raw_text})
                messages_model.append({
                    "role": "user",
                    "content": "חייב לתת לפחות קטגוריה אחת. אם אין התאמה ברורה – בחר 'אחר'. החזר אך ורק בפורמט המבני שנדרש."
                })

                if attempt < retries:
                    continue
                else:
                    return Output(explanation="", categories=["אחר"], tag="ניטרלי")

            return output
        
        except BadRequestError as e:
            if "content_filter" in str(e):
                return Output(
                    explanation="",
                    categories=['אחר'],
                    tag='ניטרלי'
                )
            raise e
        except ContentFilterFinishReasonError as e:
            return Output(
                explanation="",
                categories=['אחר'],
                tag='ניטרלי'
            )

        except RateLimitError as e:
            if e.code == 429:
                print("Rate Limit Error:", e)
                time.sleep(delay)
                continue
            else:
                raise e
        except InternalServerError as e:
            print("Internal Server Error:", e)
            time.sleep(delay)
            continue
        
    if update_progress:
        update_progress()

    return Output(
        explanation="",
        categories=["אחר"],
        tag='ניטרלי'
    )

def classify_data_azure(data, classes, num_thread=12, retries=2, delay=5):
    if not classes:
        raise ValueError("No classes provided for classification.")

    with ThreadPool(num_thread) as pool:
        results = pool.starmap(get_classes_azure, [(val, classes, retries, delay) for val in data])

    return [r.categories if r else None for r in results], [r.tag if r else None for r in results]