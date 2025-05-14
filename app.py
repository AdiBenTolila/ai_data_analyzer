import streamlit as st
import pandas as pd
from streamlit_tags import st_tags
import altair as alt
from models import first_classification_ai, classify_data, first_classification_ai_gpt, classify_data_gpt

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
    if df is not None:
        st.success("File uploaded and read successfully!")
        st.write("Here's a preview:")
        st.dataframe(df.head())

        # 1. Initialize the regenerate counter once
        if "regenerate_count" not in st.session_state:
            st.session_state["regenerate_count"] = 0

        # 2. Multiselect input
        columns_to_classify = st.multiselect("Select one or more columns to classify:", df.columns, key=f'{st.session_state["last_uploaded_file"]}_{st.session_state["last_selected_sheet"]}_selectCols')
        if columns_to_classify:
            # 3. Regenerate button
            if st.button("🔁 Regenerate Suggestions"):
                if not columns_to_classify:
                    st.error("Please select at least one column to classify.")
                else:
                    st.session_state["regenerate_count"] += 1  # Change input to cached function
                    try:
                        suggested = first_classification_ai(df, columns_to_classify)
                        # suggested = first_classification_ai_gpt(df, columns_to_classify)
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

            # Step 3: Classify
            count = 0
            if classes and columns_to_classify and st.button("🚀 Run Classification"):    
                df = df.head(50)
                df = df.dropna(subset=columns_to_classify)
                
                with st.spinner("Classifying...", show_time=True):
                    data_to_classify = df[columns_to_classify].dropna()
                    classifications_indexes = ~df[columns_to_classify].isna().any(axis=1)
                    
                    result = classify_data(data_to_classify.values, classes, num_thread=4)
                    # result = classify_data_gpt(data_to_classify.squeeze().tolist(), classes, num_thread=16)

                    df["AI סיווג"] = pd.Series([None] * len(df), dtype=object)
                    df.loc[classifications_indexes, 'AI סיווג'] = pd.Series(result, dtype=object)
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