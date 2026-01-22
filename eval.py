import pandas as pd
from models import get_classes_azure, classify_data_azure
from openai import OpenAI, RateLimitError, InternalServerError, ContentFilterFinishReasonError
from multiprocessing.pool import ThreadPool
from typing import List, Literal
from pydantic import BaseModel
from pydantic import Field
from openai import AzureOpenAI
from openai import BadRequestError

import streamlit as st
az_client = AzureOpenAI(
    api_version=st.secrets["azure"]["api_version"],
    azure_endpoint=st.secrets["azure"]["endpoint"],
    api_key=st.secrets["azure"]["api_key"],
)

# =============================
# CONFIG
# =============================
RAW_FILE = "ai_classifier.xlsx"
LABELED_FILE = "human_classifier.xlsx"
 
ID_COL = "ID"
TEXT_COL = "Comment"

HUMAN_CAT_COL = "Human_Categories"
SYSTEM_CAT_COL = "AI_Categories"
 
HUMAN_TAG_COL = "Human_Tag"
SYSTEM_TAG_COL = "AI_Tag"

DELIMITER = ","
 
#the best 
# top_p=0 -> temperature=0.6 ->
# Accuracy (all or nothing):  0.7000 (70.00%)
# Accuracy (at least one):  0.8600 (86.00%)
# Precision: 0.7900 (79.00%)
# Recall:    0.8200 (82.00%)
# F1 Score:  0.8047 (80.47%)
# Tags Accuracy:  0.9200 (92.00%)

# top_p=0 -> temperature=0 ->
# Accuracy (all or nothing):  0.7400 (74.00%)
# Accuracy (at least one):  0.9200 (92.00%)
# Precision: 0.8300 (83.00%)
# Recall:    0.8800 (88.00%)
# F1 Score:  0.8543 (85.43%)
# Tags Accuracy:  0.8800 (88.00%)

# temperature:  0  top_p:  0.6
#  50 rows
# Accuracy (all or nothing):  0.7400 (74.00%)
# Accuracy (at least one):  0.9200 (92.00%)
# Precision: 0.8300 (83.00%)
# Recall:    0.8700 (87.00%)
# F1 Score:  0.8495 (84.95%)
# Tags Accuracy:  0.8800 (88.00%)

# temperature:  0.8  top_p:  0.3
#  50 rows
# Accuracy (all or nothing):  0.7200 (72.00%)
# Accuracy (at least one):  0.8600 (86.00%)
# Precision: 0.7900 (79.00%)
# Recall:    0.8200 (82.00%)
# F1 Score:  0.8047 (80.47%)
# Tags Accuracy:  0.9400 (94.00%)

# temperature:  0.6  top_p:  0
# all rows, second 4
# Accuracy (all or nothing):  0.7087 (70.87%)
# Accuracy (at least one):  0.8937 (89.37%)
# Precision: 0.8241 (82.41%)
# Recall:    0.8255 (82.55%)
# F1 Score:  0.8248 (82.48%)
# Tags Accuracy:  0.8858 (88.58%)

# temperature:  0.6  top_p:  0
# all rows, first 4
# Accuracy (all or nothing):  0.6969 (69.69%)
# Accuracy (at least one):  0.9134 (91.34%)
# Precision: 0.8261 (82.61%)
# Recall:    0.8570 (85.70%)
# F1 Score:  0.8412 (84.12%)
# Tags Accuracy:  0.8543 (85.43%)

# temperature:  0  top_p:  0 
# all rows, first 4 , no top p
# Accuracy (all or nothing):  0.7126 (71.26%)
# Accuracy (at least one):  0.9055 (90.55%)
# Precision: 0.8313 (83.13%)
# Recall:    0.8451 (84.51%)
# F1 Score:  0.8382 (83.82%)
# Tags Accuracy:  0.8858 (88.58%)

# temperature:  0.6  top_p:  0
# all rows, first 4 , with top p
# Accuracy (all or nothing):  0.7008 (70.08%)
# Accuracy (at least one):  0.9094 (90.94%)
# Precision: 0.8248 (82.48%)
# Recall:    0.8491 (84.91%)
# F1 Score:  0.8367 (83.67%)
# Tags Accuracy:  0.8661 (86.61%)

# temperature:  0  top_p:  0
# all rows, second 4 , no top p, with prompt_text
# Accuracy (all or nothing):  0.7402 (74.02%)
# Accuracy (at least one):  0.9252 (92.52%)
# Precision: 0.8584 (85.84%)
# Recall:    0.8615 (86.15%)
# F1 Score:  0.8600 (86.00%)
# Tags Accuracy:  0.8740 (87.40%)

# temperature:  0  top_p:  0 
# 100 rows, second 4 , no top p, with prompt_text old nuturel

# Classified 0/100 rows
# Classified 54/100 rows
# ==================================================
# PART A: EVALUATION METRICS CATEGORIES
# ==================================================
# Accuracy (all or nothing):  0.7600 (76.00%)
# Accuracy (at least one):  0.9000 (90.00%)
# Precision: 0.8400 (84.00%)
# Recall:    0.8600 (86.00%)
# F1 Score:  0.8499 (84.99%)

# ======================================================================
# PART B: TAG EVALUATION (SINGLE-LABEL)
# ======================================================================
# Tags Accuracy:  0.8200 (82.00%)
temperature = [0]
top_ps = [0]
# classes =["אחר", "בעיות באתר האינטרנט", "בעיות טכניות","זמני המתנה","טיפול בערעורים","טיפול בבקשות","שירות טלפוני","שירות נציגים","שירות כללי","מידע ותשובות"]
classes =["אחר","איכות וטעם האוכל", "מגוון מנות ותפריט","תור וזמן המתנה","גודל וכמות המנות","שירות ואדיבות","מרוצה מהמסעדה","סלטים ותוספות","ניקיון והיגיינה","מחירים ותשלומים","מוזיקה ורעש"]
for temp in temperature:
    for topP in top_ps:
        print("#######################################################################################################################################")
        print("temperature: ",temp," top_p: ",topP,"\n100 rows, second 4 , no top p, with prompt_text\n")
        raw_df = pd.read_excel(RAW_FILE)
        raw_df = raw_df.head(100)

        system_categories = []
        system_tags = []
        for i, text in enumerate(raw_df[TEXT_COL]):
            if not isinstance(text, str) or not text.strip():
                system_categories.append("")
                system_tags.append(None)
                continue
            batch = {i: text}

            # cats = raw_df.loc[i, SYSTEM_CAT_COL]
            # tag = raw_df.loc[i, SYSTEM_TAG_COL]
            result = get_classes_azure(texts=batch, classes=classes, retries=2, delay=5, client=az_client, update_progress=None,temperature = temp , top_p = topP)#(text)
            cats = result.categories
            tag = result.tag

            # system_categories.append(cats)
            if isinstance(cats,list):
                system_categories.append(DELIMITER.join(cats))
            else:
                system_categories.append(cats)
            system_tags.append(tag)
            # print("cats",cats)
            if i % 54 == 0:
                print(f"Classified {i}/{len(raw_df)} rows")
        
        raw_df[SYSTEM_CAT_COL] = system_categories
        raw_df[SYSTEM_TAG_COL] = system_tags
        # =============================
        # STEP 2 – LOAD LABELED EXCEL
        # =============================
        human_df = pd.read_excel(LABELED_FILE)
        human_df = human_df.head(100)

        raw_df[ID_COL] = raw_df.index
        human_df[ID_COL] = human_df.index

        # =============================
        # STEP 3 – MERGE
        # =============================
        df = raw_df.merge(
            human_df[
                [ID_COL, HUMAN_CAT_COL, HUMAN_TAG_COL]
            ],
            on=ID_COL,
            how="inner"
        )
        df.to_excel("classified_data.xlsx")
        # print("df\n",df[::-1])
        # =============================
        # PART A – CATEGORY EVALUATION (MULTI-LABEL)
        # =============================
        y_true_cat = df[HUMAN_CAT_COL].fillna("").apply(
            lambda x: [c.strip() for c in x.split(DELIMITER) if c.strip()]
        )
        
        y_pred_cat = df[SYSTEM_CAT_COL].fillna("").apply(
            lambda x: [c.strip() for c in x.split(DELIMITER) if c.strip()]
        )
        # Get all unique categories from both human and system labels
        all_categories = set(classes)
        
        # """
        #     Recall: For a specific example, how many of the true labels also appeared in the prediction.
        #     Precision: For a specific example, how many of the predicted categories actually appeared in the true labeling,
        #     and we averaged this over all examples.
        #     Accuracy: Out of the entire table, what percentage of the predictions were correct (all or nothing).
        # """
        # Calculate TP, FP, FN, TN for each sample
        TPs = []
        FPs = []
        FNs = []
        TNs = []
        accuracies = []
        iou = []
        for i in range(len(df)):
            true_set = set(y_true_cat.iloc[i])
            pred_set = set(y_pred_cat.iloc[i])
            # print("true_set",true_set,"pred_set",pred_set)

            # True Positives: categories predicted AND actually present
            TPs.append(len(true_set & pred_set))
            # False Positives: categories predicted but NOT actually present
            FPs.append(len(pred_set - true_set))
            # False Negatives: categories actually present but NOT predicted
            FNs.append(len(true_set - pred_set))
            # True Negatives: categories neither predicted nor actually present
            # For each sample, TN = total_categories - (TP + FP + FN for that sample)
            sample_tp = len(true_set & pred_set)
            sample_fp = len(pred_set - true_set)
            sample_fn = len(true_set - pred_set)
            TNs.append(len(all_categories) - (sample_tp + sample_fp + sample_fn))

            accuracies.append(true_set == pred_set)
            if sample_tp >= 1:
                iou.append(1)
        conf_df = pd.DataFrame({'tp':TPs, 'tn':TNs, 'fp':FPs, 'fn':FNs, 'accuracy':accuracies})
        pd.concat([df, conf_df], axis=1).to_excel("confusion.xlsx")

        # accuracy = ((conf_df.tp + conf_df.tn) / (conf_df.tp + conf_df.tn + conf_df.fp + conf_df.fn)).mean()
        accuracy = conf_df.accuracy.mean()
        iou_acc = len(iou)/len(df)
        precision = (conf_df.tp / (conf_df.tp + conf_df.fp)).mean()
        recall = (conf_df.tp / (conf_df.tp + conf_df.fn)).mean()
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Print metrics
        print("=" * 50)
        print("PART A: EVALUATION METRICS CATEGORIES")
        # print("prompt:CLASSIFY_TEXT_PROMPT_ORG_V3_HE,row numbers:100")
        # print("\ntemperature",0,"no top_p")
        print("=" * 50)
        print(f"Accuracy (all or nothing):  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Accuracy (at least one):  {iou_acc:.4f} ({iou_acc*100:.2f}%)")
        print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1 Score:  {f1_score:.4f} ({f1_score*100:.2f}%)")
        print()

        # =============================
        # PART B – TAG EVALUATION (SINGLE-LABEL)
        # =============================
        print("=" * 70)
        print("PART B: TAG EVALUATION (SINGLE-LABEL)")
        print("=" * 70)
        
        # Get human and system tags
        y_true_tag = df[HUMAN_TAG_COL].fillna("").astype(str).str.strip()
        y_pred_tag = df[SYSTEM_TAG_COL].fillna("").astype(str).str.strip()

        tag_accuracy =  (y_true_tag == y_pred_tag).mean()
        print(f"Tags Accuracy:  {tag_accuracy:.4f} ({tag_accuracy*100:.2f}%)")
