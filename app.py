import streamlit as st
import pandas as pd
from streamlit_tags import st_tags
import altair as alt
from models import first_classification_ai_azure, classify_data_azure
import io

def filter_dataframe_by_categories_and_tags(df, selected_categories, selected_tags):
    """Filter dataframe to show only rows that contain at least one of the selected categories"""
    mask = pd.Series([True] * len(df), index=df.index)  # Start with all True
    if not selected_categories and not selected_tags:
        return mask
    
    if selected_categories:
        # Create a boolean mask for rows that contain any of the selected categories
        mask = df['AI סיווג'].apply(lambda x: any(cat in x if isinstance(x, list) else False for cat in selected_categories)) & mask
    if selected_tags:
        # Create a boolean mask for rows that contain any of the selected tags
        mask = df['AI תגים'].apply(lambda x: x in selected_tags) & mask
    return mask
st.logo("logo.svg",size = "large", link="https://www.jerusalem.muni.il/")
st.markdown("""
    <style>
    @media only screen and (min-width: 600px) {
        .stMainBlockContainer {
            max-width: 90%;
            margin: 0 auto;
        }
    }
    img[alt="Logo"] {
        height: 3.5rem !important;
    }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Mashov AI", layout="wide", page_icon=":bar_chart:", initial_sidebar_state="expanded")
try:
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
        if "filter_mask" in st.session_state:
            del st.session_state['filter_mask']
        if "last_uploaded_file" in st.session_state:
            del st.session_state["last_uploaded_file"]
        if 'dataframe' in st.session_state:
            del st.session_state["dataframe"]
        if 'classified_data' in st.session_state:
            del st.session_state['classified_data']
            st.session_state['is_classified']=False
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
                st.session_state['is_classified']=False
            if "filter_mask" in st.session_state:
                del st.session_state['filter_mask']
        
        if uploaded_file.name.endswith('.xlsx'):  # Only for Excel
            if "last_selected_sheet" not in st.session_state or selected_sheet != st.session_state["last_selected_sheet"]:
                st.session_state["suggested_categories"] = []
                st.session_state["regenerate_count"] = 0
                st.session_state["category_filter"] = []
                st.session_state["tags_filter"] = []
                st.session_state["last_selected_sheet"] = selected_sheet
                if "filter_mask" in st.session_state:
                    del st.session_state['filter_mask']
    else:
        st.info("**Important Note:** The system uses AI. Please ensure that no sensitive or private information that could compromise user privacy is uploaded.")
except Exception as e:
    st.error(f"Error processing file!")

def generate_suggestions(df, columns_to_classify):
    with st.spinner("Generating categories..."):
        if not columns_to_classify:
            st.error("Please select at least one column to classify.")
        else:
            if "classified_data" in st.session_state:
                del st.session_state["classified_data"]
                st.session_state['is_classified']=False
            if "filter_mask" in st.session_state:
                del st.session_state['filter_mask']
            st.session_state["regenerate_count"] += 1  # Change input to cached function
            try:
                suggested = first_classification_ai_azure(df, columns_to_classify)
                st.session_state["suggested_categories"] = suggested
            except Exception as e:
                st.error(f"Error generating suggestions: {e}")

with st.sidebar:
    if 'last_uploaded_file' in st.session_state:
        if df is not None:
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
                    st.session_state['is_classified']=False
                if "filter_mask" in st.session_state:
                    del st.session_state['filter_mask']
            else:
                if "selected_columns" in st.session_state and st.session_state["selected_columns"] != columns_to_classify:
                    st.session_state["suggested_categories"] = []
                    st.session_state["regenerate_count"] = 0
                    st.session_state["category_filter"] = []
                    st.session_state["tags_filter"] = []
                    if "classified_data" in st.session_state:
                        del st.session_state["classified_data"]
                        st.session_state['is_classified']=False
                    if "filter_mask" in st.session_state:
                        del st.session_state['filter_mask']

                st.session_state["selected_columns"] = columns_to_classify

                # 3. Regenerate button
                if st.button("🔁 Regenerate Suggestions"):
                    generate_suggestions(df, columns_to_classify)
                if columns_to_classify and st.session_state['regenerate_count']==0:
                    generate_suggestions(df, columns_to_classify)

                if "suggested_categories" in st.session_state and st.session_state["suggested_categories"] is not None:
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
                        invalid_classes = [c for c in classes if not classes]
                        if invalid_classes:
                            st.error(f"Invalid category names: {', '.join(invalid_classes)}. Please use only letters, numbers, spaces, commas, and parentheses.")
                    if classes != st.session_state["suggested_categories"]:
                        st.session_state["suggested_categories"] = classes
                        if "classified_data" in st.session_state:
                            del st.session_state["classified_data"]
                            st.session_state['is_classified']=False
                        if "filter_mask" in st.session_state:
                            del st.session_state['filter_mask']

                    prompt_text = st.text_area(
                        label = "AI instructions", 
                        help= "If you want to add instructions to the AI model, add them here\n\n recommendation: explain the model what the file is about", 
                        placeholder= "Write here...", 
                        width= "stretch"
                    )
                else:
                    classes = None
                    prompt_text = ""
                count = 0
                if classes and columns_to_classify and not invalid_classes and st.button("🚀 Run Classification"):    
                    df.dropna(subset=columns_to_classify, inplace=True, ignore_index=True)
                    
                    with st.spinner("Classifying...", show_time=True):
                        data_to_classify = df[columns_to_classify]
                        categories, tags = classify_data_azure(data_to_classify.to_dict(orient='records'), classes,prompt_text)

                        df['AI סיווג'] = categories
                        df['AI תגים'] = tags
                    st.session_state["classified_data"] = df
                    st.session_state["original_columns"] = df.columns.tolist()
                    st.session_state['is_classified']=True
                    if "edit_toggle" in st.session_state:
                        st.session_state['edit_toggle'] = False
                    st.success("Classification completed!")
                    st.rerun()
if df is not None and columns_to_classify and 'last_uploaded_file' in st.session_state and "classified_data" in st.session_state:
    table_tab, plot_tab = st.tabs(["Table", "Statistics"])
    # Get all unique categories from the classified data
    with table_tab:
        all_categories_in_data = set()
        for cat_list in st.session_state["classified_data"]['AI סיווג']:
            if isinstance(cat_list, list):
                all_categories_in_data.update(cat_list)
        
        st.session_state.setdefault("category_filter", [])
        st.session_state.setdefault("tags_filter", [])

        category_options = sorted(list(classes))
        tags_options = ['מרוצה', 'לא מרוצה', 'ניטרלי']

        category_default = [v for v in st.session_state.get("category_filter", []) if v in category_options]
        tags_default = [v for v in st.session_state.get("tags_filter", []) if v in tags_options]

        # Filter selection
        filter_col1, filter_col2 = st.columns([0.5, 0.5])
        with filter_col1:
            category_filter = st.multiselect(
                "Select categories to display (leave empty to show all):",
                options=sorted(classes),
                default=category_default,
                key="category_filter_select"
            )
        with filter_col2:
            tags_filter = st.multiselect(
                "Select tags to display (leave empty to show all):",
                options=tags_options,
                default=tags_default,
                key="tags_filter_select"
            )

        # Update session state
        st.session_state["category_filter"] = category_filter
        st.session_state["tags_filter"] = tags_filter

        def apply_edit():
            if "classified_data" in st.session_state:
                mask = st.session_state.get('filter_mask', pd.Series([True] * len(df), index=df.index))
                idx_map = st.session_state["classified_data"].index[mask]
                diff = st.session_state.get(f"editable_df", {})
                for _idx, changes in diff.get('edited_rows',{}).items():
                    row_idx = idx_map[_idx]
                    for col, val in changes.items():
                        st.session_state["classified_data"].at[row_idx,col]=val

        df = st.session_state["classified_data"]

        # Apply category filter
        if category_filter or tags_filter:
            mask = filter_dataframe_by_categories_and_tags(df, st.session_state["category_filter"], st.session_state["tags_filter"])
        else:
            mask = pd.Series([True] * len(df), index=df.index)
        if ("classified_data" in st.session_state) and (st.session_state["classified_data"] is not None) and not (mask == st.session_state.get('filter_mask', pd.Series([True] * len(df), index=df.index))).all():
            apply_edit()
            mask = filter_dataframe_by_categories_and_tags(df, st.session_state["category_filter"], st.session_state["tags_filter"])
        st.session_state['filter_mask'] = mask
        cols_first = columns_to_classify.copy()
        if "AI סיווג" in df.columns:
            cols_first.append("AI סיווג")

        if "AI תגים" in df.columns:
            cols_first.append("AI תגים")

        # All other columns (not in the first group)
        remaining_cols = [col for col in df.columns if col not in cols_first]

        # Final order
        reordered_cols = cols_first + remaining_cols

        all_valid_categories = set(st.session_state.get("suggested_categories", []))
        color_palette = [
                    "#F3E2BC", "#E9D7FF", "#DAF3E0", "#F1D6D6", 
                    "#ADE5F1D3", "#C7C4C4", "#AEC4FF", "#C8CAEC", "#E9B5A8",
                    "#BDD4E9", "#EEEEEE", "#ECE1A4", "#DAB2EE", "#F6F7CA",
                    "#BAE9E9", "#FFB8DE", "#CBFAFDDF", "#A7DDC7", "#DBFAF4",
        ]

        edit = st.toggle("Edit", value=False, on_change=apply_edit, key='edit_toggle') 
        df_filtered = df[mask]
        edited_df = st.data_editor(
            df_filtered, 
            key=f"editable_df", 
            disabled=not edit,
            num_rows='fixed', 
            width='stretch', 
            hide_index=True,
            row_height = 50 ,
            column_order=reordered_cols,
            column_config={
                **{
                    col:None for col in df_filtered.columns if col not in columns_to_classify
                },
                'AI סיווג': st.column_config.MultiselectColumn(
                    "AI סיווג",
                    help = "בחר קטגוריות מהרשימה",
                    options = sorted(list(all_valid_categories)),
                    color = color_palette,
                    width = "large"
                ),
                'AI תגים': st.column_config.SelectboxColumn(
                    "AI תגים",
                    options=['מרוצה', 'לא מרוצה','ניטרלי'],
                    required=True,
                )
            }
        )
        # Show filtered count
        col2,col1 = st.columns([1,1])
        with col1: 
            if category_filter or tags_filter:
                total_rows = len(st.session_state["classified_data"])
                filtered_rows = len(df_filtered)
                st.caption(f"Page  {filtered_rows} out of {total_rows}", text_alignment='right')
        with col2:
            full_df = st.session_state["classified_data"]  # Use full dataset
            exploded_df = full_df.explode('AI סיווג')['AI סיווג']
            chart_data = exploded_df.value_counts().reset_index()
            chart_data.columns = ["קטגוריה", "כמות"]

            all_categories = pd.Series(classes, name="קטגוריה")
            chart_data_full = pd.merge(
                all_categories.to_frame(), chart_data, on="קטגוריה", how="left"
            )
            chart_data_full["כמות"] = chart_data_full["כמות"].fillna(0).astype(int)
            chart_data_full = chart_data_full.sort_values(by="כמות", ascending=False).reset_index(drop=True)
            def download_excel(columns=st.session_state["original_columns"]):
                full_excel_buffer = io.BytesIO()
                with pd.ExcelWriter(full_excel_buffer, engine="xlsxwriter", engine_kwargs={'options': {'nan_inf_to_errors': True}}) as writer:#TODO
                    df_export = full_df.copy()
                    
                    # Join lists into comma-separated strings for export
                    def _join_if_iterable(x):
                        if isinstance(x, (list, tuple, set)):
                            return ", ".join(map(str, x))
                        return x
                    
                    if "AI סיווג" in df_export.columns:
                        df_export["AI סיווג"] = df_export["AI סיווג"].apply(_join_if_iterable)
                    
                    # Write main data sheet
                    df_export[columns].to_excel(writer, index=False, sheet_name="Classified Data")
                    
                    # Get workbook and worksheet objects
                    workbook = writer.book
                    worksheet_data = writer.sheets["Classified Data"]
                    
                    # Find column indices for AI סיווג and AI תגים
                    col_classification = columns.index("AI סיווג") if "AI סיווג" in columns else None
                    col_tags = columns.index("AI תגים") if "AI תגים" in columns else None
                    
                    # Convert column indices to Excel column letters
                    def col_to_letter(col_idx):
                        """Convert 0-based column index to Excel letter (0=A, 1=B, etc.)"""
                        letter = ''
                        col_idx += 1  # Excel is 1-based
                        while col_idx > 0:
                            col_idx -= 1
                            letter = chr(col_idx % 26 + 65) + letter
                            col_idx //= 26
                        return letter
                    
                    classification_col_letter = col_to_letter(col_classification) if col_classification is not None else None
                    tags_col_letter = col_to_letter(col_tags) if col_tags is not None else None
                    
                    # Data starts at row 2 (row 1 has headers)
                    data_start_row = 2
                    data_end_row = len(df_export) + 1
                    
                    # Create Frequency sheet
                    worksheet_freq = workbook.add_worksheet('Frequency')
                    all_categories = classes
                    all_tags = full_df['AI תגים'].unique().tolist() if 'AI תגים' in full_df.columns else []
                    # Format for headers
                    header_format = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC'})
                    
                    # ===== SECTION 1: Category Distribution =====
                    worksheet_freq.write('A1', 'קטגוריה', header_format)
                    worksheet_freq.write('B1', 'כמות', header_format)
                    worksheet_freq.set_column('A:A', 30)
                    worksheet_freq.set_column('B:B', 10)
                    category_counts = full_df.explode('AI סיווג')['AI סיווג'].value_counts()
                    for idx, category in enumerate(all_categories, start=2):
                            worksheet_freq.write(f'A{idx}', category)
                            formula = f'=COUNTIF(\'Classified Data\'!${classification_col_letter}${data_start_row}:${classification_col_letter}${data_end_row},"*{category}*")'
                            count = category_counts.get(category,0)
                            worksheet_freq.write(f'B{idx}', formula, None, count)
                    
                    cat_end_row = len(all_categories) + 1
                    
                    # Convert to Excel Table
                    worksheet_freq.add_table(f'A1:B{cat_end_row}', {
                        'name': 'CategoryTable',
                        'columns': [
                            {'header': 'קטגוריה'},
                            {'header': 'כמות'}
                        ],
                        'style': 'Table Style Light 9'
                    })
                    
                    worksheet_freq.conditional_format(f'B2:B{cat_end_row}', {
                        'type': '3_color_scale',
                        'min_color': "#FFC7CE",
                        'mid_color': "#FFEB9C",
                        'max_color': "#C6EFCE"
                    })
                    
                    # Chart 1: Column chart - Category distribution
                    chart1 = workbook.add_chart({'type': 'column'})
                    chart1.add_series({
                        'name': 'שכיחויות',
                        'categories': f'=Frequency!$A$2:$A${cat_end_row}',
                        'values': f'=Frequency!$B$2:$B${cat_end_row}',
                        'fill': {'color': '#5DADE2'}
                    })
                    chart1.show_blanks_as('zero')#TODO
                    chart1.set_legend({'none': True})
                    chart1.set_title({'name': 'התפלגות קטגוריות'})
                    chart1.set_x_axis({'name': 'קטגוריה'})
                    chart1.set_y_axis({'name': 'כמות'})
                    chart1.set_style(10)
                    worksheet_freq.insert_chart('H2', chart1, {'x_scale': 1.5, 'y_scale': 1.5})
                    
                    # ===== SECTION 2: Tags Distribution =====
                    start_row_tags = cat_end_row + 15
                    worksheet_freq.write(start_row_tags, 0, 'תג', header_format)
                    worksheet_freq.write(start_row_tags, 1, 'כמות', header_format)
                    
                    # Write tags and create COUNTIF formulas
                    tag_counts = full_df['AI תגים'].value_counts()
                    for idx, tag in enumerate(all_tags, start=start_row_tags + 1):
                        worksheet_freq.write(idx, 0, tag)
                        formula = f'=COUNTIF(\'Classified Data\'!${tags_col_letter}${data_start_row}:${tags_col_letter}${data_end_row},"{tag}")'
                        worksheet_freq.write(idx, 1, formula, None, tag_counts[tag])
                    
                    tags_end_row = start_row_tags + len(all_tags) + 1
                    
                    # Chart 2: Pie chart - Tags distribution
                    chart3 = workbook.add_chart({'type': 'pie'})
                    chart3.add_series({
                        'name': 'התפלגות תגים',
                        'categories': f'=Frequency!$A${start_row_tags+2}:$A${tags_end_row}',
                        'values': f'=Frequency!$B${start_row_tags+2}:$B${tags_end_row}',
                    })
                    chart3.show_blanks_as('zero')#TODO
                    chart3.set_title({'name': 'התפלגות תגים'})
                    chart3.set_style(10)
                    worksheet_freq.insert_chart('H27', chart3, {'x_scale': 1.5, 'y_scale': 1.5})
                    
                    # ===== SECTION 3: Category by Tags (tab2 graph) =====
                    # Create a new sheet for the detailed breakdown
                    worksheet_breakdown = workbook.add_worksheet('Category-Tags Breakdown')
                    
                    worksheet_breakdown.set_column('A:A', 30)
                    worksheet_breakdown.set_column('B:B', 20)
                    worksheet_breakdown.set_column('C:C', 10)
                    
                    # Simpler approach: Create a pivot-like structure with FORMULAS
                    # Write pivot data starting from column A (for chart 4 - Categories by Tags)
                    pivot_start_col = 0  # Column A (0-indexed: 0)
                    worksheet_breakdown.write(0, pivot_start_col, 'קטגוריה', header_format)
                    
                    for tag_idx, tag in enumerate(all_tags):
                        worksheet_breakdown.write(0, pivot_start_col + tag_idx + 1, tag, header_format)
                    
                    # Write categories and FORMULAS to count occurrences
                    category_tag_counts = full_df.explode('AI סיווג').groupby(['AI סיווג', 'AI תגים']).size()
                    for cat_idx, category in enumerate(all_categories, start=1):
                        worksheet_breakdown.write(cat_idx, pivot_start_col, category)
                        for tag_idx, tag in enumerate(all_tags):
                            col_letter = col_to_letter(pivot_start_col + tag_idx + 1)
                            # COUNTIFS formula to count rows where both category AND tag match
                            formula = f'=SUMPRODUCT((ISNUMBER(SEARCH("{category}",\'Classified Data\'!${classification_col_letter}${data_start_row}:${classification_col_letter}${data_end_row})))*(\'Classified Data\'!${tags_col_letter}${data_start_row}:${tags_col_letter}${data_end_row}="{tag}"))'
                            count = category_tag_counts.get((category,tag), 0)
                            worksheet_breakdown.write(cat_idx, pivot_start_col + tag_idx + 1, formula, None, count)
                    
                    pivot_end_row = len(all_categories) + 1
                    pivot_end_col_letter = col_to_letter(pivot_start_col + len(all_tags))
                    pivot_start_col_letter = col_to_letter(pivot_start_col)
                    
                    # Add conditional formatting to the data range
                    if len(all_tags) > 0:
                        data_range_start = col_to_letter(pivot_start_col + 1)
                        data_range_end = col_to_letter(pivot_start_col + len(all_tags))
                        worksheet_breakdown.conditional_format(f'{data_range_start}2:{data_range_end}{pivot_end_row}', {
                            'type': '3_color_scale',
                            'min_color': "#FFC7CE",
                            'mid_color': "#FFEB9C",
                            'max_color': "#C6EFCE"
                        })
                    
                    # Chart 4: Stacked column - Categories distributed by tags
                    chart4 = workbook.add_chart({'type': 'column', 'subtype': 'stacked'})
                    tag_colors = {            
                        'מרוצה': "#9DEEC6",      
                        'לא מרוצה': "#E9B2AE",   
                        'ניטרלי': "#F0E9AA"}        
                    for tag_idx, tag in enumerate(all_tags):            
                        col_letter = col_to_letter(pivot_start_col + tag_idx + 1)            
                        series_config = {                
                            'name': f'=\'Category-Tags Breakdown\'!${col_letter}$1',                
                            'categories': f'=\'Category-Tags Breakdown\'!${pivot_start_col_letter}$2:${pivot_start_col_letter}${pivot_end_row}',                
                            'values': f'=\'Category-Tags Breakdown\'!${col_letter}$2:${col_letter}${pivot_end_row}',            
                        }            
                        # Add color if tag is in our color mapping            
                        if tag in tag_colors:                
                            series_config['fill'] = {'color': tag_colors[tag]}            
                        chart4.add_series(series_config)
                    chart4.set_title({'name': 'התפלגות קטגוריות לפי תגים'})
                    chart4.set_x_axis({'name': 'קטגוריה'})
                    chart4.set_y_axis({'name': 'כמות'})
                    chart4.set_style(10)
                    
                    # Position chart to the right of the data (with more space)
                    chart_col = col_to_letter(len(all_tags) + 5)  # Added more columns for spacing
                    worksheet_breakdown.insert_chart(f'{chart_col}2', chart4, {'x_scale': 1.5, 'y_scale': 1.5})
                    
                    # Chart 5: Stacked column - Tags distributed by categories (tab3)
                    # Create another pivot with tags as rows and categories as columns
                    pivot2_start_row = pivot_end_row + 13
                    pivot2_start_col = 0  # Column A
                    
                    worksheet_breakdown.write(pivot2_start_row, pivot2_start_col, 'תג', header_format)
                    
                    for cat_idx, category in enumerate(all_categories):
                        worksheet_breakdown.write(pivot2_start_row, pivot2_start_col + cat_idx + 1, category, header_format)
                    
                    # Write tags and FORMULAS to count occurrences
                    for tag_idx, tag in enumerate(all_tags, start=1):
                        worksheet_breakdown.write(pivot2_start_row + tag_idx, pivot2_start_col, tag)
                        for cat_idx, category in enumerate(all_categories):
                            col_letter = col_to_letter(pivot2_start_col + cat_idx + 1)
                            # COUNTIFS formula to count rows where both tag AND category match
                            formula = f'=SUMPRODUCT((\'Classified Data\'!${tags_col_letter}${data_start_row}:${tags_col_letter}${data_end_row}="{tag}")*(ISNUMBER(SEARCH("{category}",\'Classified Data\'!${classification_col_letter}${data_start_row}:${classification_col_letter}${data_end_row}))))'
                            count = category_tag_counts.get((category,tag), 0)
                            worksheet_breakdown.write(pivot2_start_row + tag_idx, pivot2_start_col + cat_idx + 1, formula, None, count)
                    
                    pivot2_end_row = pivot2_start_row + len(all_tags)
                    pivot2_start_col_letter = col_to_letter(pivot2_start_col)
                    
                    # Add conditional formatting to the second data range
                    if len(all_categories) > 0:
                        data_range_start = col_to_letter(pivot2_start_col + 1)
                        data_range_end = col_to_letter(pivot2_start_col + len(all_categories))
                        worksheet_breakdown.conditional_format(f'{data_range_start}{pivot2_start_row + 2}:{data_range_end}{pivot2_end_row + 1}', {
                            'type': '3_color_scale',
                            'min_color': "#FFC7CE",
                            'mid_color': "#FFEB9C",
                            'max_color': "#C6EFCE"
                        })
                    
                    chart5 = workbook.add_chart({'type': 'column', 'subtype': 'stacked'})
                    
                    for cat_idx, category in enumerate(all_categories):
                        col_letter = col_to_letter(pivot2_start_col + cat_idx + 1)
                        chart5.add_series({
                            'name': f'=\'Category-Tags Breakdown\'!${col_letter}${pivot2_start_row + 1}',
                            'categories': f'=\'Category-Tags Breakdown\'!${pivot2_start_col_letter}${pivot2_start_row + 2}:${pivot2_start_col_letter}${pivot2_end_row + 1}',
                            'values': f'=\'Category-Tags Breakdown\'!${col_letter}${pivot2_start_row + 2}:${col_letter}${pivot2_end_row + 1}',
                        })
                    
                    chart5.set_title({'name': 'התפלגות תגים לפי קטגוריות'})
                    chart5.set_x_axis({'name': 'תג'})
                    chart5.set_y_axis({'name': 'כמות'})
                    chart5.set_style(10)
                    
                    # Position chart to the right of the second data table (with more space)
                    chart_col2 = col_to_letter(len(all_categories) + 2)  # Added more columns for spacing
                    worksheet_breakdown.insert_chart(f'{chart_col2}{pivot2_start_row + 1}', chart5, {'x_scale': 1.5, 'y_scale': 1.5})
                    
                return full_excel_buffer.getvalue()


            st.download_button(
                    label="⤓",
                    data=download_excel,
                    disabled=edit,
                    file_name="classified_data_and_distribution.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    with plot_tab:
        if edit == True:
            st.warning("Please note that you are currently in edit mode. The data shown reflects the state before your changes were saved.")

        full_df = st.session_state["classified_data"]  # Use full dataset
        exploded_df = full_df.explode('AI סיווג')['AI סיווג']
        chart_data = exploded_df.value_counts().reset_index()
        chart_data.columns = ["קטגוריה", "כמות"]

        tags_count_per_category = full_df.explode('AI סיווג').groupby(['AI סיווג', 'AI תגים']).size().reset_index(name='count')

        st.header("📈 התפלגות הסיווגים לפי קטגוריה")
        tab1, tab2, tab3, tab4, tab5= st.tabs(["📊 התפלגות קטגוריות", "📊 התפלגות קטגוריות לפי תגים", "📊 התפלגות תגים לפי קטגוריות", "📊 התפלגות קטגוריות (Pie Chart)", "📊 התפלגות תגים (Pie Chart)"])
        with tab1:
            chart = alt.Chart(chart_data).mark_bar(color="#72b3d8ff",opacity=0.6).encode(
                x=alt.X("קטגוריה:N",sort="-y", axis=alt.Axis(labelAngle=0, labelExpr="split(datum.label, ' ')")),
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
            tags_chart = alt.Chart(tags_count_per_category).mark_bar(opacity=0.6).encode(
                x=alt.X('AI סיווג:N', title='קטגוריות',sort='-y', axis=alt.Axis(labelAngle=0, labelExpr="split(datum.label, ' ')")),
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
            cats_py_tags_chart = alt.Chart(tags_count_per_category).mark_bar(opacity=0.6).encode(
                x=alt.X('AI תגים:N', title='תגים', sort='-y', axis=alt.Axis(labelAngle=0)),
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
            pie_chart = alt.Chart(chart_data).mark_arc(innerRadius=60,opacity=0.6).encode(
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
            tags_pie_chart = alt.Chart(tags_count).mark_arc(innerRadius=60,opacity=0.6).encode(
                theta=alt.Theta("כמות:Q", stack=True),
                color=alt.Color("תג:N", legend=alt.Legend(title="תגים")),
                tooltip=["תג:N", "כמות:Q"]
            ).properties(
                width=400,
                height=400,
                title="התפלגות תגים (Pie Chart)"
            )
            st.altair_chart(tags_pie_chart)
elif df is not None and 'last_uploaded_file' in st.session_state:
    st.write("Preview:")
    st.dataframe(df, hide_index=True,row_height = 50)
