import streamlit as st
import pandas as pd
import numpy as np
import faiss
import openai
import re
import os
from PIL import Image
from sklearn.model_selection import train_test_split

# 페이지 설정
st.set_page_config(
    page_title="AI 위험성평가 자동 생성 및 사고 예측",
    page_icon="🛠️",
    layout="wide"
)

# 스타일 적용
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .info-text {
        font-size: 1rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #e3f2fd;
        padding: 5px;
        border-radius: 5px;
    }
    .result-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-top: 10px;
        margin-bottom: 10px;
        border-left: 5px solid #4CAF50;
    }
    .phase-badge {
        background-color: #4CAF50;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

# 헤더 표시
st.markdown('<div class="main-header">AI 활용 위험성평가 자동 생성 및 사고 예측</div>', unsafe_allow_html=True)

# 세션 상태 초기화
if "index" not in st.session_state:
    st.session_state.index = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "retriever_pool_df" not in st.session_state:
    st.session_state.retriever_pool_df = None

# 탭 설정
tabs = st.tabs(["시스템 개요", "위험성 평가 (Phase 1)", "개선대책 생성 (Phase 2)"])

# ------------------ 유틸리티 함수 ------------------

def determine_grade(value):
    if 16 <= value <= 25:
        return 'A'
    elif 10 <= value <= 15:
        return 'B'
    elif 5 <= value <= 9:
        return 'C'
    elif 3 <= value <= 4:
        return 'D'
    elif 1 <= value <= 2:
        return 'E'
    else:
        return '알 수 없음'


def load_data(selected_dataset_name):
    try:
        df = pd.read_excel(f"{selected_dataset_name}.xlsx")
        if '삭제 Del' in df.columns:
            df = df.drop(['삭제 Del'], axis=1)
        df = df.iloc[1:]
        df = df.rename(columns={df.columns[4]: '빈도', df.columns[5]: '강도'})
        df['T'] = pd.to_numeric(df.iloc[:,4]) * pd.to_numeric(df.iloc[:,5])
        df = df.iloc[:,:7]
        df.rename(
            columns={
                '작업활동 및 내용\nWork & Contents':'작업활동 및 내용',
                '유해위험요인 및 환경측면 영향\nHazard & Risk':'유해위험요인 및 환경측면 영향',
                '피해형태 및 환경영향\nDamage & Effect':'피해형태 및 환경영향'
            }, inplace=True)
        df = df.rename(columns={df.columns[6]:'T'})
        df['등급'] = df['T'].apply(determine_grade)
        return df
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {str(e)}")
        st.warning("Excel 파일을 찾을 수 없어 샘플 데이터를 생성합니다.")
        data = {
            "작업활동 및 내용":["Shoring Installation","In and Out of materials","Transport / Delivery","Survey and Inspection"],
            "유해위험요인 및 환경측면 영향":["Fall and collision due to unstable ground","Overturning of transport vehicle","Collision between transport vehicle","Personnel fall while inspecting"],
            "피해형태 및 환경영향":["Injury","Equipment damage","Collision injury","Fall injury"],
            "빈도":[3,3,3,2],"강도":[2,3,5,3]
        }
        df = pd.DataFrame(data)
        df['T'] = df['빈도']*df['강도']
        df['등급'] = df['T'].apply(determine_grade)
        return df


def embed_texts_with_openai(texts, model="text-embedding-3-large", api_key=None):
    if api_key:
        openai.api_key = api_key
    embeddings = []
    progress_bar = st.progress(0)
    total = len(texts)
    for idx, text in enumerate(texts):
        try:
            text = str(text).replace("\n"," ")
            response = openai.Embedding.create(model=model, input=[text])
            embeddings.append(response["data"][0]["embedding"])
        except:
            embeddings.append([0]*1536)
        progress_bar.progress((idx+1)/total)
    return embeddings


def generate_with_gpt(prompt, api_key=None, model="gpt-4o"):
    if api_key:
        openai.api_key = api_key
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role":"system","content":"위험성 평가 및 개선대책 생성을 돕는 도우미입니다."},
                      {"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=250
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        st.error(f"GPT API 호출 중 오류 발생: {str(e)}")
        return None

# ----- Phase1 전용 Prompt/Parser (유해위험요인 예측) -----

def construct_prompt_phase1_for_hazard(retrieved_docs, query_activity):
    prompt = ""
    for i, row in enumerate(retrieved_docs.itertuples(),1):
        activity = getattr(row,'content')
        hazard = getattr(row,'유해위험요인 및 환경측면 영향')
        prompt += f"예시 {i}:\n입력: {activity}\n출력: {hazard}\n\n"
    prompt += (
        f"입력: {query_activity}\n"
        "위 작업활동 및 내용을 바탕으로 유해위험요인을 한 문장으로 예측하세요.\n"
        "다음 JSON 형식으로 반환하세요:\n"
        '{"유해위험요인":"여기에 예측 결과"}\n'
    )
    return prompt


def parse_gpt_output_phase1_for_hazard(gpt_output):
    try:
        m = re.search(r'\{.*\}', gpt_output, re.DOTALL)
        if not m:
            return None
        data = re.json.loads(m.group())
        return data.get("유해위험요인")
    except:
        return None

# ----- Phase2: 개선대책 생성 관련 함수 -----
def compute_rrr(T_before, T_after):
    if T_before == 0:
        return 0.0
    return ((T_before - T_after) / T_before) * 100.0


def construct_prompt_phase2(retrieved_docs, activity_text, hazard_text, freq, intensity, T, target_language="Korean"):
    example_section = ""
    examples_added = 0
    for _, row in retrieved_docs.iterrows():
        try:
            improvement_plan = ""
            for field in ['개선대책 및 세부관리방안','개선대책','개선방안']:
                if field in row and pd.notna(row[field]):
                    improvement_plan = row[field]
                    break
            if not improvement_plan:
                continue
            orig_f = int(row['빈도'])
            orig_i = int(row['강도'])
            orig_T = orig_f * orig_i
            imp_f, imp_i, imp_T = 1,1,1
            for пат in [('개선 후 빈도','개선 후 강도','개선 후 T'),('개선빈도','개선강도','개선T')]:
                if all(p in row for p in пат):
                    imp_f = int(row[пат[0]]); imp_i = int(row[пат[1]]); imp_T = int(row[пат[2]]); break
            example_section += (
                "Example:\n"
                f"Input (Activity): {row['작업활동 및 내용']}\n"
                f"Input (Hazard): {row['유해위험요인 및 환경측면 영향']}\n"
                f"Input (Original Frequency): {orig_f}\n"
                f"Input (Original Intensity): {orig_i}\n"
                f"Input (Original T): {orig_T}\n"
                "Output (Improvement Plan and Risk Reduction) in JSON:\n"
                "{\n"
                f"  \"개선대책\": \"{improvement_plan}\",\n"
                f"  \"개선 후 빈도\": {imp_f},\n"
                f"  \"개선 후 강도\": {imp_i},\n"
                f"  \"개선 후 T\": {imp_T},\n"
                f"  \"T 감소율\": {compute_rrr(orig_T, imp_T):.2f}\n"
                "}\n\n"
            )
            examples_added += 1
            if examples_added >= 3:
                break
        except:
            continue
    if examples_added == 0:
        example_section = "... 기본 예시 ..."
    prompt = (
        f"{example_section}"
        "Now here is a new input
