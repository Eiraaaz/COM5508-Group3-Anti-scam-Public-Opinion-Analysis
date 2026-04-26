import pandas as pd
import json
from openai import OpenAI
from tqdm import tqdm
import time
from typing import Dict, Any

# ==================== Configuration ====================
API_KEY = "sk-9d18he6N6PDGzCEVBTMbJuZhM4mfYnbxFDgPC2wmS9LIRWsc" 

client = OpenAI(
    api_key=API_KEY,
    base_url="https://sg.uiuiapi.com/v1"
)

MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.0
MAX_RETRIES = 3
# ============================================================

def create_extraction_prompt(text: str) -> str:
    
    prompt = f"""You are a professional fraud public opinion analyst.
Carefully analyze the provided comment content and **dynamically infer** the most appropriate categories based solely on the context, tone, described events, implied meanings, and overall semantics.
Do NOT restrict yourself to any predefined lists or examples — generate the categories that best fit the text.

Strictly output ONLY a valid JSON object in the exact format below. No extra text.

{{
  "fraud_types": ["the most suitable fraud type(s) inferred from this text"] or []   // can be multiple, completely dynamic
  "victim_demographics": {{
    "gender": "the most appropriate gender inferred or unknown",
    "age_group": "the most appropriate age group inferred or unknown",
    "occupation": "the most appropriate occupation inferred or unknown",
    "other_identity": "any notable additional victim identity feature or null"
  }},
  "main_theme_stage": "the most appropriate discussion stage inferred from context (one concise phrase)",
}}

comment content: {text}
"""
    return prompt


def llm_structured_extract(text: str) -> Dict[str, Any]:
    if not isinstance(text, str) or pd.isna(text) or len(text.strip()) < 5:
        return {
            "fraud_types": [],
            "victim_demographics": {"gender": "unknown", "age_group": "unknown", "occupation": "unknown", "other_identity": None},
            "main_theme_stage": "other",
        }
    
    prompt = create_extraction_prompt(text)
    
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            result.setdefault("fraud_types", [])
            result.setdefault("victim_demographics", {"gender": "unknown", "age_group": "unknown", "occupation": "unknown", "other_identity": None})
            result.setdefault("main_theme_stage", "other")
            
            return result
            
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    
    print("LLM extraction failed, returning default values")
    return {
        "fraud_types": [],
        "victim_demographics": {"gender": "unknown", "age_group": "unknown", "occupation": "unknown", "other_identity": None},
        "main_theme_stage": "other",
    }

# ==================== Main ====================
def run_comment_extraction(comments_df: pd.DataFrame) -> pd.DataFrame:
    
    comments_structured = comments_df.copy()
    tqdm.pandas(desc="Comment Extraction")
    comments_structured['structured_comment'] = comments_structured['content'].progress_apply(
        llm_structured_extract
    )
    
    comments_structured = pd.concat([
        comments_structured.drop(columns=['structured_comment']),
        comments_structured['structured_comment'].apply(pd.Series)
    ], axis=1)
    
    victim_demo = comments_structured['victim_demographics'].apply(pd.Series)
    victim_demo.columns = [f"victim_{col}" for col in victim_demo.columns]
    comments_structured = pd.concat([comments_structured.drop(columns=['victim_demographics']), victim_demo], axis=1)
    
    print("Completed.")
    return comments_structured

comments_df = pd.read_csv("comments_cleaned_2025.csv", encoding="utf-8")
comments_structured = run_comment_extraction(comments_df)
comments_structured.to_csv("comments_structured_result.csv", index=False, encoding="utf-8-sig")