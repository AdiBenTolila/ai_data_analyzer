import streamlit as st
import pandas as pd
from streamlit_tags import st_tags
import altair as alt
from models import first_classification_ai, classify_data, first_classification_ai_gpt, classify_data_gpt
import io
import re

def extract_classes(s, valid_classes):
    # Sort valid classes by length (longest first) to prioritize greedy match
    sorted_classes = sorted(valid_classes, key=len, reverse=True)
    
    found = []
    remaining = s

    while remaining:
        matched = False
        for cls in sorted_classes:
            # Try to match class at start of the string (with optional surrounding spaces)
            pattern = r'^\s*' + re.escape(cls) + r'\s*(,|$)'
            m = re.match(pattern, remaining)
            if m:
                found.append(cls)
                # Remove matched portion including trailing comma
                remaining = remaining[m.end():]
                matched = True
                break
        if not matched:
            # Unable to match anything — stop or raise an error
            break
    
    return found

def cat_str_to_lst(cat_str, classes):
    if isinstance(cat_str, str):
        return extract_classes(cat_str, classes)
    elif isinstance(cat_str, list):
        return [c.strip() for c in cat_str if c.strip()]
    return cat_str

def cat_lst_to_str(cat_lst, classes):
    if isinstance(cat_lst, list):
        return ', '.join(cat_lst)
    elif isinstance(cat_lst, str):
        matches = extract_classes(cat_lst, classes)
        if matches:
            return ', '.join([c.strip() for c in matches if c.strip()])
        else:
            return ''
    return cat_lst

st.markdown("""
    <style>
    @media only screen and (min-width: 600px) {
        .stMainBlockContainer {
            max-width: 70%;
            margin: 0 auto;
        }
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Excel Q&A Assistant")
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "csv"])

df = None
selected_sheet = None

# If a file is uploaded
if uploaded_file is None or uploaded_file.name != st.session_state.get("last_uploaded_file"):
    st.session_state["suggested_categories"] = []
    st.session_state["regenerate_count"] = 0
    st.session_state["last_selected_sheet"] = None
    if "last_uploaded_file" in st.session_state:
        del st.session_state["last_uploaded_file"]
    if 'dataframe' in st.session_state:
        del st.session_state["dataframe"]
    if 'classified_data' in st.session_state:
        del st.session_state['classified_data']
    df = None
if st.session_state.get("dataframe") is not None:
    df = st.session_state["dataframe"]
elif uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        selected_sheet = "CSV file"
        with st.spinner("Reading file...", show_time=True):
            df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
            st.session_state["dataframe"] = df

    else:
        excel_data = pd.ExcelFile(uploaded_file)
        sheet_names = ["Select a sheet..."] + excel_data.sheet_names
        selected_sheet = st.selectbox("Choose a sheet", sheet_names, key=f"sheet_select_{uploaded_file.name}")

        if selected_sheet != "Select a sheet...":
            with st.spinner("Reading file...", show_time=True):
                df = pd.read_excel(uploaded_file, sheet_name=selected_sheet, skiprows=0)
                st.session_state["dataframe"] = df

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
        if not columns_to_classify:
            st.session_state["suggested_categories"] = []
            st.session_state["regenerate_count"] = 0
            st.session_state["selected_columns"] = []
            if "classified_data" in st.session_state:
                del st.session_state["classified_data"]
        else:
            if "selected_columns" in st.session_state and st.session_state["selected_columns"] != columns_to_classify:
                st.session_state["suggested_categories"] = []
                st.session_state["regenerate_count"] = 0
                if "classified_data" in st.session_state:
                    del st.session_state["classified_data"]

            st.session_state["selected_columns"] = columns_to_classify
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
                    key=f"tag-input-{st.session_state['regenerate_count']}",
                )
                classes = [c.strip() for c in classes if c.strip()]
                # handle invalid category names
                if classes:
                    invalid_classes = [c for c in classes if not re.match(r'^[\w\s(),]+$', c)]
                    if invalid_classes:
                        st.error(f"Invalid category names: {', '.join(invalid_classes)}. Please use only letters, numbers, spaces, commas, and parentheses.")
                if classes != st.session_state["suggested_categories"]:
                    st.session_state["suggested_categories"] = classes
                    if "classified_data" in st.session_state:
                        del st.session_state["classified_data"]
            else:
                classes = []

            # Step 3: Classify
            count = 0
            if classes and columns_to_classify and not invalid_classes and st.button("🚀 Run Classification"):    
                # df = df.head(200)
                df = df.dropna(subset=columns_to_classify)
                
                with st.spinner("Classifying...", show_time=True):
                    data_to_classify = df[columns_to_classify]
                    
                    result = classify_data(data_to_classify.to_dict(orient='records'), classes, num_thread=4)

                    df['AI סיווג'] = result
                st.session_state["classified_data"] = df
                st.session_state["original_columns"] = df.columns.tolist()
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
                df_display['AI סיווג'] = df_display['AI סיווג'].apply(lambda x: cat_lst_to_str(x, classes))

                # Ensure 'AI סיווג' and selected columns exist
                cols_first = columns_to_classify.copy()
                if "AI סיווג" in df_display.columns:
                    cols_first.append("AI סיווג")

                # All other columns (not in the first group)
                remaining_cols = [col for col in df_display.columns if col not in cols_first]

                # Final order
                reordered_cols = cols_first + remaining_cols

                
                if edit:
                    st.session_state["in_edit_mode"] = True
                    all_valid_categories = set(st.session_state.get("suggested_categories", []))

                    escaped = sorted([cls.replace('(', r'\(').replace(')', r'\)') for cls in all_valid_categories], key=len, reverse=True)
                    group = '|'.join(escaped)
                    pattern = fr"^\s*({group})(\s*,\s*({group}))*\s*$"
                    edited_df = st.data_editor(df_display[reordered_cols], key="editable_df", num_rows="dynamic", use_container_width=True, hide_index=True,column_config={
                            'AI סיווג': st.column_config.TextColumn(
                                "AI סיווג",
                                validate=fr"^\s*({group})(\s*,\s*({group}))*\s*$",
                            )
                        })
                    edited_df['AI סיווג'] = edited_df['AI סיווג'].apply(lambda x: cat_lst_to_str(x, classes))
                    st.session_state["edited_df"] = edited_df
                else:
                    if "in_edit_mode" in st.session_state and st.session_state["in_edit_mode"] and "edited_df" in st.session_state:
                        edited_df = st.session_state["edited_df"]
                        if edited_df is not None and not edited_df.equals(df_display):
                        # Convert back to list
                            edited_df['AI סיווג'] = edited_df['AI סיווג'].apply(lambda x: cat_str_to_lst(x, classes))
                            st.session_state["classified_data"] = edited_df
                        del st.session_state["in_edit_mode"]
                        del st.session_state["edited_df"]
                        st.rerun()
                    # Prepare reordered display: [selected columns] + [AI סיווג] + [rest]
                    df_display = st.session_state["classified_data"]
                    # Show table in the desired order
                    st.dataframe(df_display[reordered_cols])


            if "AI סיווג" in df.columns and st.session_state.get("classified_data") is not None:
                exploded_df = df.explode('AI סיווג')['AI סיווג']
                chart_data = exploded_df.value_counts().reset_index()
                chart_data.columns = ["קטגוריה", "כמות"]
                col1, col2 = st.columns([0.9, 0.1])
                with col1:
                    st.header("📈 התפלגות הסיווגים לפי קטגוריה")
                with col2:
                    all_categories = pd.Series(classes, name="קטגוריה")
                    chart_data_full = pd.merge(
                        all_categories.to_frame(), chart_data, on="קטגוריה", how="left"
                    )
                    chart_data_full["כמות"] = chart_data_full["כמות"].fillna(0).astype(int)
          
                    full_excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(full_excel_buffer, engine="xlsxwriter") as writer:

                        df_export = df.copy()
                        df_export['AI סיווג'] = df_export['AI סיווג'].apply(lambda x: cat_lst_to_str(x, classes))
                        df_export[st.session_state["original_columns"]].to_excel(writer, index=False, sheet_name="Classified Data")
                        # Add another sheet for the frequency chart
                        chart_data_full.to_excel(writer, index=False, sheet_name="Frequency", startrow=1, header=False)

                        workbook = writer.book
                        worksheet = writer.sheets["Frequency"]

                        # titles
                        header_format = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC'})
                        worksheet.write('A1', 'קטגוריה', header_format)
                        worksheet.write('B1', 'כמות', header_format)
                        worksheet.set_column('A:A', 30)
                        worksheet.set_column('B:B', 10)

                        # colorful formatting
                        worksheet.conditional_format(f'B2:B{len(chart_data_full)+1}', {
                            'type': '3_color_scale',
                            'min_color': "#FFC7CE",
                            'mid_color': "#FFEB9C",
                            'max_color': "#C6EFCE"
                        })

                        # column chart
                        chart = workbook.add_chart({'type': 'column'})
                        chart.add_series({
                            'name':       'שכיחויות',
                            'categories': f'=Frequency!$A$2:$A${len(chart_data_full)+1}',
                            'values':     f'=Frequency!$B$2:$B${len(chart_data_full)+1}',
                            'fill':       {'color': '#5DADE2'}
                        })
                        chart.set_title({'name': 'התפלגות קטגוריות'})
                        chart.set_x_axis({'name': 'קטגוריה'})
                        chart.set_y_axis({'name': 'כמות'})
                        chart.set_style(10)
                        worksheet.insert_chart('D2', chart, {'x_scale': 1.5, 'y_scale': 1.5})

                    if not ("in_edit_mode" in st.session_state and st.session_state["in_edit_mode"] and "edited_df" in st.session_state):
                        st.download_button(
                                label="📥הורד",
                                data=full_excel_buffer.getvalue(),
                                file_name="classified_data_and_distribution.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                chart = alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X("קטגוריה:N", sort="-y"),
                    y=alt.Y("כמות:Q"),
                    tooltip=["קטגוריה", "כמות"]
                ).properties(
                    width=600,
                    height=400
                )
                st.altair_chart(chart)

