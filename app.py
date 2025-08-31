import streamlit as st
import pandas as pd
from streamlit_tags import st_tags
import altair as alt
from models import first_classification_ai, classify_data, first_classification_ai_gpt, classify_data_gpt, first_classification_ai_azure, classify_data_azure
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

def filter_dataframe_by_categories_and_tags(df, selected_categories, selected_tags):
    """Filter dataframe to show only rows that contain at least one of the selected categories"""
    if not selected_categories and not selected_tags:
        return df
    
    mask = pd.Series([True] * len(df))  # Start with all True
    if selected_categories:
        # Create a boolean mask for rows that contain any of the selected categories
        mask = df['AI סיווג'].apply(lambda x: any(cat in x if isinstance(x, list) else False for cat in selected_categories)) & mask
    if selected_tags:
        # Create a boolean mask for rows that contain any of the selected tags
        mask = df['AI תגים'].apply(lambda x: x in selected_tags) & mask
    return df[mask]

def save_edited_data():
    if "edited_df" in st.session_state:
        edited_df = st.session_state["edited_df"]
        if edited_df is not None:
            # Convert back to list and update the original dataset
            edited_df_copy = edited_df.copy()
            edited_df_copy['AI סיווג'] = edited_df_copy['AI סיווג'].apply(lambda x: cat_str_to_lst(x, classes))
            
            # Update the original classified_data with the changes
            filtered_indices = st.session_state.get("filtered_indices", [])
            
            # Ensure we have valid indices and matching lengths
            if len(filtered_indices) == len(edited_df_copy):
                for i, original_idx in enumerate(filtered_indices):
                    try:
                        # Update AI classification
                        st.session_state["classified_data"].at[original_idx, 'AI סיווג'] = edited_df_copy.iloc[i]['AI סיווג']
                        
                        # Update other columns that might have been edited
                        for col in edited_df_copy.columns:
                            if col != 'AI סיווג' and col in st.session_state["classified_data"].columns:
                                st.session_state["classified_data"].at[original_idx, col] = edited_df_copy.iloc[i][col]
                    except (KeyError, IndexError) as e:
                        st.error(f"Error updating row {i}: {e}")
                        continue
            else:
                st.error(f"Index mismatch: {len(filtered_indices)} filtered indices vs {len(edited_df_copy)} edited rows")

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
st.set_page_config(page_title="Mashov AI", layout="wide", page_icon=":bar_chart:")
st.title("📊 Mashov AI")
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "csv"])

df = None
selected_sheet = None

# If a file is uploaded
if uploaded_file is None or uploaded_file.name != st.session_state.get("last_uploaded_file"):
    st.session_state["suggested_categories"] = []
    st.session_state["regenerate_count"] = 0
    st.session_state["last_selected_sheet"] = None
    st.session_state["category_filter"] = []
    st.session_state["tags_filter"] = []
    if "last_uploaded_file" in st.session_state:
        del st.session_state["last_uploaded_file"]
    if 'dataframe' in st.session_state:
        del st.session_state["dataframe"]
    if 'classified_data' in st.session_state:
        del st.session_state['classified_data']
    df = None
if st.session_state.get("dataframe") is not None:
    df = st.session_state["dataframe"]

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        selected_sheet = "CSV file"
        with st.spinner("Reading file...", show_time=True):
            df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
            st.session_state["dataframe"] = df

    else:
        excel_data = pd.ExcelFile(uploaded_file)
        sheet_names = excel_data.sheet_names
        if len(sheet_names) > 1:
            selected_sheet = st.selectbox("Choose a sheet", sheet_names, key=f"sheet_select_{uploaded_file.name}", index=None)

            if not selected_sheet is None:
                with st.spinner("Reading file...", show_time=True):
                    df = pd.read_excel(uploaded_file, sheet_name=selected_sheet, skiprows=0)
                    st.session_state["dataframe"] = df
            else:
                if 'dataframe' in st.session_state:
                    del st.session_state["dataframe"]
                    df = None

        elif len(sheet_names) == 1:
            df = pd.read_excel(uploaded_file, sheet_name=sheet_names[0], skiprows=0)
            st.session_state["dataframe"] = df
        else:
            st.error("The uploaded file does not contain any sheets.")

    if "last_uploaded_file" not in st.session_state or uploaded_file.name != st.session_state["last_uploaded_file"]:
        st.session_state["suggested_categories"] = []
        st.session_state["regenerate_count"] = 0
        st.session_state["category_filter"] = []
        st.session_state["tags_filter"] = []
        st.session_state["last_uploaded_file"] = uploaded_file.name
        if 'classified_data' in st.session_state:
            del st.session_state['classified_data']

    
    if uploaded_file.name.endswith('.xlsx'):  # Only for Excel
        if "last_selected_sheet" not in st.session_state or selected_sheet != st.session_state["last_selected_sheet"]:
            st.session_state["suggested_categories"] = []
            st.session_state["regenerate_count"] = 0
            st.session_state["category_filter"] = []
            st.session_state["tags_filter"] = []
            st.session_state["last_selected_sheet"] = selected_sheet
if 'last_uploaded_file' in st.session_state:
    if df is not None:
        # st.success("File uploaded and read successfully!")
        st.write("Here's a preview:")
        st.dataframe(df, height=250, hide_index=True)

        # 1. Initialize the regenerate counter once
        if "regenerate_count" not in st.session_state:
            st.session_state["regenerate_count"] = 0

        # 2. Multiselect input
        columns_to_classify = st.multiselect("Select one or more columns to classify:", df.columns, key=f'{st.session_state["last_uploaded_file"]}_{st.session_state["last_selected_sheet"]}_selectCols')
        if not columns_to_classify:
            st.session_state["suggested_categories"] = []
            st.session_state["regenerate_count"] = 0
            st.session_state["selected_columns"] = []
            st.session_state["category_filter"] = []
            st.session_state["tags_filter"] = []
            if "classified_data" in st.session_state:
                del st.session_state["classified_data"]
        else:
            if "selected_columns" in st.session_state and st.session_state["selected_columns"] != columns_to_classify:
                st.session_state["suggested_categories"] = []
                st.session_state["regenerate_count"] = 0
                st.session_state["category_filter"] = []
                st.session_state["tags_filter"] = []
                if "classified_data" in st.session_state:
                    del st.session_state["classified_data"]

            st.session_state["selected_columns"] = columns_to_classify
            # 3. Regenerate button
            if st.button("🔁 Regenerate Suggestions"):
                if not columns_to_classify:
                    st.error("Please select at least one column to classify.")
                else:
                    if "classified_data" in st.session_state:
                        del st.session_state["classified_data"]
                    st.session_state["regenerate_count"] += 1  # Change input to cached function
                    try:
                        suggested = first_classification_ai_azure(df, columns_to_classify)
                        st.session_state["suggested_categories"] = suggested
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
                    invalid_classes = [c for c in classes if not re.match(r"^[\w\s(),']+$", c)]
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
                    
                    categories, tags = classify_data_azure(data_to_classify.to_dict(orient='records'), classes)

                    df['AI סיווג'] = categories
                    df['AI תגים'] = tags
                st.session_state["classified_data"] = df
                st.session_state["original_columns"] = df.columns.tolist()
                st.success("Classification completed!")

            if "classified_data" in st.session_state:

                # Get all unique categories from the classified data
                all_categories_in_data = set()
                for cat_list in st.session_state["classified_data"]['AI סיווג']:
                    if isinstance(cat_list, list):
                        all_categories_in_data.update(cat_list)

                
                col1, col2 = st.columns([0.9, 0.1])                
                with col2:
                    edit = st.toggle("Edit", value=False)

                df = st.session_state["classified_data"]

                # Filter selection
                filter_col1, filter_col2 = st.columns([0.5, 0.5])
                with filter_col1:
                    category_filter = st.multiselect(
                        "Select categories to display (leave empty to show all):",
                        options=sorted(list(all_categories_in_data)),
                        default=st.session_state.get("default_category_filter", []),
                        key="category_filter_select"
                    )
                with filter_col2:
                    tags_filter = st.multiselect(
                        "Select tags to display (leave empty to show all):",
                        options=['מרוצה', 'לא מרוצה', 'ניטרלי'],
                        default=st.session_state.get("default_tags_filter", []),
                        key="tags_filter_select"
                    )

                if ('category_filter' in st.session_state and category_filter != st.session_state["category_filter"]) \
                    or ('tags_filter' in st.session_state and tags_filter != st.session_state["tags_filter"]) or \
                    (category_filter !=[] and 'category_filter' not in st.session_state) or \
                    (tags_filter !=[] and 'tags_filter' not in st.session_state):
                    save_edited_data()
                # Update session state
                st.session_state["category_filter"] = category_filter
                st.session_state["tags_filter"] = tags_filter

                # Apply category filter
                if category_filter or tags_filter:
                    df_filtered = filter_dataframe_by_categories_and_tags(df, st.session_state["category_filter"], st.session_state["tags_filter"])
                else:
                    df_filtered = df.copy()

                # Prepare display dataframe
                df_display = df_filtered.copy()

                # Ensure 'AI סיווג' and selected columns exist
                cols_first = columns_to_classify.copy()
                if "AI סיווג" in df_display.columns:
                    cols_first.append("AI סיווג")

                if "AI תגים" in df_display.columns:
                    cols_first.append("AI תגים")

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
                    
                    # Store the original indices to maintain connection with full dataset
                    # Use the actual DataFrame index, not a list conversion
                    st.session_state["filtered_indices"] = df_display.index.tolist()
                    
                    # Convert to string format for editing
                    df_edit_display = df_display.copy()
                    df_edit_display['AI סיווג'] = df_edit_display['AI סיווג'].apply(lambda x: cat_lst_to_str(x, classes))
                    
                    # Reset index for editor but keep track of original indices
                    df_for_editor = df_edit_display[reordered_cols].reset_index(drop=True)
                    
                    edited_df = st.data_editor(
                        df_for_editor, 
                        key=f"editable_df_{len(category_filter)}_{hash(tuple(category_filter))}", 
                        num_rows='fixed', 
                        use_container_width=True, 
                        hide_index=True,
                        
                        column_config={
                            **{
                                col:None for col in df_for_editor.columns if col not in columns_to_classify
                            },
                            'AI סיווג': st.column_config.TextColumn(
                                "AI סיווג",
                                validate=fr"^\s*({group})(\s*,\s*({group}))*\s*$",
                            ),
                            'AI תגים': st.column_config.SelectboxColumn(
                                "AI תגים",
                                options=['מרוצה', 'לא מרוצה','ניטרלי'],
                                required=True,
                            )
                        }
                    )
                    
                    st.session_state["edited_df"] = edited_df
                else:
                    if "in_edit_mode" in st.session_state and st.session_state["in_edit_mode"] and "edited_df" in st.session_state:
                        save_edited_data()
                        # Clean up session state
                        del st.session_state["in_edit_mode"]
                        del st.session_state["edited_df"]
                        if "filtered_indices" in st.session_state:
                            del st.session_state["filtered_indices"]
                        st.rerun()
                    
                    # Show table in the desired order
                    st.dataframe(df_display[reordered_cols],
                                column_config={
                                    **{
                                        col:None for col in df_display.columns if col not in columns_to_classify
                                    },
                                    'AI סיווג': st.column_config.ListColumn(
                                        "AI סיווג",
                                        help="סיווגים שנוצרו על ידי ה-AI",
                                    ),
                                    'AI תגים': st.column_config.ListColumn(
                                        "AI תגים",
                                        help="תגים שנוצרו על ידי ה-AI",
                                    )
                                }
)

                # Show filtered count
                if category_filter:
                    total_rows = len(st.session_state["classified_data"])
                    filtered_rows = len(df_display)
                    st.caption(f"Showing {filtered_rows} out of {total_rows} rows")

            # Use the full classified data (not filtered) for charts and downloads
            if "AI סיווג" in df.columns and st.session_state.get("classified_data") is not None:
                full_df = st.session_state["classified_data"]  # Use full dataset
                exploded_df = full_df.explode('AI סיווג')['AI סיווג']
                chart_data = exploded_df.value_counts().reset_index()
                chart_data.columns = ["קטגוריה", "כמות"]

                tags_count_per_category = full_df.explode('AI סיווג').groupby(['AI סיווג', 'AI תגים']).size().reset_index(name='count')
                col1, col2 = st.columns([0.9, 0.1])
                with col2:
                    all_categories = pd.Series(classes, name="קטגוריה")
                    chart_data_full = pd.merge(
                        all_categories.to_frame(), chart_data, on="קטגוריה", how="left"
                    )

                    chart_data_full["כמות"] = chart_data_full["כמות"].fillna(0).astype(int)
                    chart_data_full = chart_data_full.sort_values(by="כמות", ascending=False).reset_index(drop=True)

                    full_excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(full_excel_buffer, engine="xlsxwriter") as writer:

                        df_export = full_df.copy()  # Use full dataset for export
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
                with col1:
                    st.header("📈 התפלגות הסיווגים לפי קטגוריה")
                tab1, tab2, tab3, tab4, tab5= st.tabs(["📊 התפלגות קטגוריות", "📊 התפלגות קטגוריות לפי תגים", "📊 התפלגות תגים לפי קטגוריות", "📊 התפלגות קטגוריות (Pie Chart)", "📊 התפלגות תגים (Pie Chart)"])
                with tab1:
                    chart = alt.Chart(chart_data).mark_bar().encode(
                        x=alt.X("קטגוריה:N", sort="-y"),
                        y=alt.Y("כמות:Q"),
                        tooltip=["קטגוריה", "כמות"]
                    ).properties(
                        width=600,
                        height=400
                    )
                    st.altair_chart(chart)

                color_scale = alt.Scale(
                    domain=['מרוצה', 'לא מרוצה', 'ניטרלי'],
                    range=["#38E39E", "#ED8074", "#E5C15E"]
                )
                with tab2:
                    tags_chart = alt.Chart(tags_count_per_category).mark_bar().encode(
                        x=alt.X('AI סיווג:N', 
                                title='קטגוריות',
                                sort='-y'),  # Sort by count (descending)
                        y=alt.Y('count:Q', 
                                title='כמות'),
                        color=alt.Color('AI תגים:N', 
                                    title='תגים',
                                    scale=color_scale),
                        tooltip=['AI סיווג:N', 'AI תגים:N', 'count:Q']
                    ).properties(
                        width=600,
                        height=400,
                        title='חלוקת קטגוריות לפי תגיות'
                    )
                    st.altair_chart(tags_chart)

                with tab3:
                    cats_py_tags_chart = alt.Chart(tags_count_per_category).mark_bar().encode(
                        x=alt.X('AI תגים:N', title='תגים', sort='-y'),
                        y=alt.Y('count:Q', title='כמות'),
                        color=alt.Color('AI סיווג:N', title='קטגוריות', scale=alt.Scale(scheme='category20')),
                        tooltip=['AI תגים:N', 'AI סיווג:N', 'count:Q']
                    ).properties(
                        width=600,
                        height=400,
                        title='התפלגות תגים לפי קטגוריות'
                    )
                    st.altair_chart(cats_py_tags_chart)
                with tab4:
                    # Pie chart of category distribution (suggested chart 1)
                    pie_chart = alt.Chart(chart_data).mark_arc(innerRadius=60).encode(
                        theta=alt.Theta("כמות:Q", stack=True),
                        color=alt.Color("קטגוריה:N", legend=alt.Legend(title="קטגוריה")),
                        tooltip=["קטגוריה:N", "כמות:Q"]
                    ).properties(
                        width=400,
                        height=400,
                        title="התפלגות קטגוריות (Pie Chart)"
                    )
                    st.altair_chart(pie_chart)
                with tab5:
                    tags_count = full_df['AI תגים'].value_counts().reset_index()
                    tags_count.columns = ['תג', 'כמות']
                    tags_pie_chart = alt.Chart(tags_count).mark_arc(innerRadius=60).encode(
                        theta=alt.Theta("כמות:Q", stack=True),
                        color=alt.Color("תג:N", legend=alt.Legend(title="תגים")),
                        tooltip=["תג:N", "כמות:Q"]
                    ).properties(
                        width=400,
                        height=400,
                        title="התפלגות תגים (Pie Chart)"
                    )
                    st.altair_chart(tags_pie_chart)