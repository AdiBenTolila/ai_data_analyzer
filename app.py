import streamlit as st
import pandas as pd
from streamlit_tags import st_tags
import altair as alt
from models import first_classification_ai, classify_data, first_classification_ai_gpt, classify_data_gpt
import io

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
else:
    st.session_state["suggested_categories"] = []
    st.session_state["regenerate_count"] = 0
    st.session_state["last_selected_sheet"] = None
    st.session_state["last_uploaded_file"] = None
    st.session_state["dataframe"] = None
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
        if not columns_to_classify:
            st.session_state["suggested_categories"] = []
        else:
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
            else:
                classes = []

            # Step 3: Classify
            count = 0
            if classes and columns_to_classify and st.button("🚀 Run Classification"):    
                # df = df.head(200)
                df = df.dropna(subset=columns_to_classify)
                
                with st.spinner("Classifying...", show_time=True):
                    data_to_classify = df[columns_to_classify]
                    
                    result = classify_data(data_to_classify.values, classes, num_thread=4)

                    df['AI סיווג'] = result
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
                        df_export['AI סיווג'] = df_export['AI סיווג'].apply(
                            lambda l: ', '.join(l) if isinstance(l, list) else l
                        )
                        df_export.to_excel(writer, index=False, sheet_name="Classified Data")

                        # גיליון נוסף: טבלת שכיחויות
                        chart_data_full.to_excel(writer, index=False, sheet_name="Frequency", startrow=1, header=False)

                        workbook = writer.book
                        worksheet = writer.sheets["Frequency"]

                        # כותרות
                        header_format = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC'})
                        worksheet.write('A1', 'קטגוריה', header_format)
                        worksheet.write('B1', 'כמות', header_format)
                        worksheet.set_column('A:A', 30)
                        worksheet.set_column('B:B', 10)

                        # עיצוב צבעוני
                        worksheet.conditional_format(f'B2:B{len(chart_data_full)+1}', {
                            'type': '3_color_scale',
                            'min_color': "#FFC7CE",
                            'mid_color': "#FFEB9C",
                            'max_color': "#C6EFCE"
                        })

                        # גרף עמודות
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

                    # כפתור להורדת קובץ מלא
                    st.download_button(
                        label="📥הורד",
                        # : סיווגים + תפלגות + גרף
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