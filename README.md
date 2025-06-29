# 📊 Excel Q&A Assistant

This is a Streamlit-based interactive assistant designed to help users upload Excel or CSV files, select columns for classification, and use AI models to suggest and assign categories. It supports editing, visualizing, and downloading the results with an integrated frequency chart and analysis.

---

## 🚀 Getting Started

### 🔧 Installation

1. Clone the project:
```bash
git clone https://JerusalemMuni@dev.azure.com/JerusalemMuni/Data/_git/Surveys_AI_Classifier
cd Surveys_AI_Classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### ⚙️ Configuration

Inside the file `.streamlit/secrets.toml` configure the following fields:

#### OpenAI and Gemini API Keys

```toml
[openai]
api_key = "your-openai-key"

[[gemini]]
api_key = "gemini-key-1"

[[gemini]]
api_key = "gemini-key-2-for-threads"

[[gemini]]
api_key = "gemini-key-3-for-threads"

[[gemini]]
api_key = "gemini-key-4-for-threads"
```
---

## ▶️ How to Run

To start the Streamlit app locally, run:

```bash
streamlit run app.py
```

---

## 🧠 Features

  - Category distribution 
  - Classified data
  - Frequency chart embedded in Excel

---

## 📦 File Output

When classification is completed, you can download an Excel file that includes:
- A **Classified Data** sheet with AI labels
- A **Frequency** sheet showing category distribution
---