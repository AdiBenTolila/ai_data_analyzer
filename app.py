import streamlit as st
import pandas as pd
import io
import os
from google import genai
from streamlit_tags import st_tags
import altair as alt
import time
from typing import List, Literal


client = genai.Client(api_key=os.environ.get("API_KEY"))

client = genai.Client(api_key=gemini_key)
st.markdown("""
    <style>
    .stMainBlockContainer {
        max-width: 70%;
        margin: 0 auto;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Excel Q&A Assistant")
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "csv"])

df = None
selected_sheet = None
# If a file is uploaded
if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        selected_sheet = "CSV file"
        df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
    else:
        excel_data = pd.ExcelFile(uploaded_file)
        sheet_names = ["Select a sheet..."] + excel_data.sheet_names
        selected_sheet = st.selectbox("Choose a sheet", sheet_names, key=f"sheet_select_{uploaded_file.name}")

        if selected_sheet != "Select a sheet...":
            df = pd.read_excel(uploaded_file, sheet_name=selected_sheet, skiprows=0)

    if "last_uploaded_file" not in st.session_state or uploaded_file.name != st.session_state["last_uploaded_file"]:
        st.session_state["suggested_categories"] = []
        st.session_state["regenerate_count"] = 0
        st.session_state["last_uploaded_file"] = uploaded_file.name
        if 'classified_data' in st.session_state:
            del st.session_state['classified_data']

    
    if uploaded_file.name.endswith('.xlsx'):  # Only for Excel
        if "last_selected_sheet" not in st.session_state or selected_sheet != st.session_state["last_selected_sheet"]:
            st.session_state["suggested_categories"] = []
            st.session_state["regenerate_count"] = 0
            st.session_state["last_selected_sheet"] = selected_sheet
else:
    st.session_state["suggested_categories"] = []
    st.session_state["regenerate_count"] = 0
    st.session_state["last_selected_sheet"] = None
    st.session_state["last_uploaded_file"] = None
    if 'classified_data' in st.session_state:
        del st.session_state['classified_data']

if 'last_uploaded_file' in st.session_state:
    def first_classification_ai(columns_to_classify, classes):
        sample_text = "\n".join(df[columns_to_classify].astype(str).iloc[:, 0].dropna().head(3000).tolist())  # Take first column's sample
        prompt = f"""You are a smart data assistant.

                Below is a sample of several rows taken from a larger Excel table. Each row represents a real-world data entry.

                Your task is to analyze these rows and suggest 10 relevant and general **categories** in **Hebrew** that could be used to classify similar data in the full table.

                **Important instructions:**
                - The categories must be in **Hebrew only** (no English).
                - The last category must be "אחר" (meaning "other").

                Here are the sample rows:

                {sample_text}
                """
        response = ""
        response = client.models.generate_content(model="gemini-2.0-flash",
                                                  contents=prompt,
                                                  config={
                                                        'response_mime_type': 'application/json',
                                                        'response_schema': list[str]
                                                    }
                                                  )
        classes = response.parsed
        return classes
    


    if df is not None:
        df.columns = df.columns.map(str)
        df.fillna("", inplace=True)  # Replace all NaN with empty string
        df = df.applymap(str)  # Ensure all data is string type (useful for LLM input)
        df = df.head(20)  # Limit rows for preview
        st.success("File uploaded and read successfully!")
        st.write("Here's a preview:")
        st.dataframe(df.head())

        # 1. Initialize the regenerate counter once
        if "regenerate_count" not in st.session_state:
            st.session_state["regenerate_count"] = 0

        # 2. Multiselect input
        columns_to_classify = st.multiselect("Select one or more columns to classify:", df.columns, key=f'{st.session_state["last_uploaded_file"]}_{st.session_state["last_selected_sheet"]}_selectCols')

        # 3. Regenerate button
        if st.button("🔁 Regenerate Suggestions"):
            if not columns_to_classify:
                st.error("Please select at least one column to classify.")
            else:
                st.session_state["regenerate_count"] += 1  # Change input to cached function
                try:
                    suggested = first_classification_ai(columns_to_classify, st.session_state["regenerate_count"])
                    st.session_state["suggested_categories"] = suggested
                    st.success("New suggestions generated!")
                except Exception as e:
                    st.error(f"Error generating suggestions: {e}")
        if "suggested_categories" in st.session_state and st.session_state["suggested_categories"]:
            classes = st_tags(
                label="✏️ Choose or add categories for classification:",
                text="Press Enter to add more",
                value=st.session_state["suggested_categories"],
                suggestions=st.session_state["suggested_categories"],
                maxtags=20,
                key="tag-input-1",
            )
        else:
            classes = []
        print("classes", classes)
        
        @st.cache_data
        def get_classes(text, classes, retries=2, delay=5):
            prompt = (
                    f"You are a classification assistant. Your task is to choose the most appropriate categories from the following list:\n\n"
                    f"{classes}\n\n"
                    f"Rules:\n"
                    f"1. You may select one or more categories.\n"
                    f"2. Return them comma-separated.\n"
                    f"3. If you select ANY category other than 'אחר', then 'אחר' MUST NOT appear in your answer.\n"
                    f"4. If there are NO other matching categories at all, then return ONLY 'אחר'.\n\n"
                    f"Now, given the following text, return ONLY the category names (comma-separated), with NO explanations:\n\n"
                    f"Text: {text}"
                )
            for _ in range(retries + 1):
                try:
                    response = client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=prompt,
                        config={
                            "temperature": 0.0,
                            'response_mime_type': 'application/json',
                            'response_schema': list[Literal[tuple(classes)]],
                        }
                    )
                    return response.parsed
                except Exception as e:
                    print("error", e)
                    time.sleep(delay)
            return None

        # Step 3: Classify
        if classes and columns_to_classify and st.button("🚀 Run Classification"):    
            with st.spinner("Classifying...", show_time=True):
                df["AI סיווג"] = df[columns_to_classify].dropna().apply(
                    lambda row: get_classes(row.to_string(index=False), classes), axis=1
                )
            st.session_state["classified_data"] = df
            st.success("Classification completed!")

        if "classified_data" in st.session_state:
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.text("Classified Table:")
            with col2:
                edit = st.toggle("Edit", value=False)

            df = st.session_state["classified_data"]

            # Convert list to comma-separated strings for editing
            df_display = df.copy()
            df_display['AI סיווג'] = df_display['AI סיווג'].apply(
                lambda l: ', '.join(l) if isinstance(l, list) else l
            )

            if edit:
                edited_df = st.data_editor(df_display, key="editable_df", num_rows="dynamic", use_container_width=True, hide_index=True)

                # Save back to session_state only if edited_df is different
                if edited_df is not None and not edited_df.equals(df_display):
                    # Convert back to list
                    edited_df['AI סיווג'] = edited_df['AI סיווג'].apply(
                        lambda s: s.split(', ') if isinstance(s, str) else s
                    )
                    st.session_state["classified_data"] = edited_df
                    st.rerun()
            else:
                st.dataframe(df)

        if "AI סיווג" in df.columns:
            st.header("📈 תפלגות הסיווגים לפי קטגוריה")

            # Step 1: Split categories by comma and explode
            exploded_df = df.explode('AI סיווג')['AI סיווג']

            # Step 2: Count category frequencies
            chart_data = exploded_df.value_counts().reset_index()
            chart_data.columns = ["קטגוריה", "כמות"]

            # Step 3: Create chart
            chart = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X("קטגוריה:N", sort="-y"),
                y=alt.Y("כמות:Q"),
                tooltip=["קטגוריה", "כמות"]
            ).properties(
                width=600,
                height=400
            )

            st.altair_chart(chart)