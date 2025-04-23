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

# 유틸리티 함수
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
            "작업활동 및 내용": ["Shoring Installation","In and Out of materials","Transport / Delivery","Survey and Inspection"],
            "유해위험요인 및 환경측면 영향": ["Fall and collision due to unstable ground","Overturning of transport vehicle","Collision between transport vehicle","Personnel fall while inspecting"],
            "피해형태 및 환경영향": ["Injury","Equipment damage","Collision injury","Fall injury"],
            "빈도": [3,3,3,2], "강도": [2,3,5,3]
        }
        df = pd.DataFrame(data)
        df['T'] = df['빈도'] * df['강도']
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
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role":"system","content":"위험성 평가 및 개선대책 생성을 돕는 도우미입니다."},
            {"role":"user","content":prompt}
        ],
        temperature=0.0,
        max_tokens=250
    )
    return response['choices'][0]['message']['content'].strip()

# Phase 1: 유해위험요인 예측
def construct_prompt_phase1_for_hazard(retrieved_docs, query_activity):
    """
    retrieved_docs: DataFrame with columns 'content' and '유해위험요인 및 환경측면 영향'
    Generates a prompt listing 3 examples and the user query.
    """
    prompt = ""
    # 안전하게 iterrows를 사용하여 컬럼명 기준 접근
    for i, (_, row) in enumerate(retrieved_docs.iterrows(), 1):
        activity = row['content']
        hazard   = row['유해위험요인 및 환경측면 영향']
        # newline은 
으로 명시
        prompt += f"예시 {i}: 입력: {activity} → {hazard}
"
    # 사용자 입력 쿼리
    prompt += (
        f"입력: {query_activity}
"
        "위 작업활동 및 내용을 바탕으로 유해위험요인을 예측하세요.
"
        "JSON으로 반환: {\"유해위험요인\": \"...\"}
"
    )
    return prompt

def parse_gpt_output_phase1_for_hazard(gpt_output):
    try:
        m = re.search(r'\{.*\}', gpt_output, re.DOTALL)
        return eval(m.group())['유해위험요인'] if m else None
    except:
        return None

# Phase 2: 개선대책 생성
def compute_rrr(T_before, T_after):
    return ((T_before - T_after) / T_before * 100) if T_before else 0.0

def construct_prompt_phase2(retrieved_docs, activity_text, hazard_text, freq, intensity, T, target_language="Korean"):
    example_section = ""
    for idx, row in enumerate(retrieved_docs.itertuples(),1):
        plan = row._asdict().get('개선대책') or ''
        example_section += (
            f"Example {idx}:\n"
            f"Activity: {row._asdict()['작업활동 및 내용']}\n"
            f"Hazard: {row._asdict()['유해위험요인 및 환경측면 영향']}\n"
            f"Plan: {plan}\n\n"
        )
        if idx >= 3: break

    prompt = (
        f"{example_section}"
        "Now here is a new input:\n"
        f"Activity: {activity_text}\n"
        f"Hazard: {hazard_text}\n"
        f"Original Freq: {freq}, Intensity: {intensity}, T: {T}\n\n"
        "위 JSON 키로 개선대책 생성:\n"
        "{\n"
        "  \"개선대책\": [...],\n"
        "  \"개선 후 빈도\": NUM,\n"
        "  \"개선 후 강도\": NUM,\n"
        "  \"개선 후 T\": NUM,\n"
        "  \"T 감소율\": NUM\n"
        "}\n"
    )
    return prompt


def parse_gpt_output_phase2(gpt_output):
    try:
        m = re.search(r'\{.*\}', gpt_output, re.DOTALL)
        return eval(m.group()) if m else None
    except:
        return None

# 데이터셋 옵션
dataset_options = {
    "SWRO 건축공정 (건축)":"SWRO 건축공정 (건축)",
    "Civil (토목)":"Civil (토목)",
    "Marine (토목)":"Marine (토목)",
    "SWRO 기계공사 (플랜트)":"SWRO 기계공사 (플랜트)",
    "SWRO 전기작업표준 (플랜트)":"SWRO 전기작업표준 (플랜트)"
}

# 시스템 개요 탭
with tabs[0]:
    st.markdown('<div class="sub-header">시스템 개요</div>', unsafe_allow_html=True)
    st.markdown("LLM 기반 위험성 평가 및 개선대책 자동화 시스템입니다.")

# Phase1 탭
with tabs[1]:
    st.markdown('<div class="sub-header">위험성 평가 (Phase 1)</div>', unsafe_allow_html=True)
    api_key = st.text_input("OpenAI API 키를 입력하세요:", type="password")
    dataset = st.selectbox("데이터셋 선택", list(dataset_options.keys()))
    if st.button("데이터 로드 및 인덱스 구성"):
        if api_key:
            df = load_data(dataset_options[dataset])
            train_df, _ = train_test_split(df, test_size=0.1, random_state=42)
            retriever = train_df.copy()
            retriever['content'] = retriever['작업활동 및 내용']
            emb = embed_texts_with_openai(retriever['content'].tolist(), api_key=api_key)
            index = faiss.IndexFlatL2(len(emb[0]))
            index.add(np.array(emb, dtype='float32'))
            st.session_state.index = index
            st.session_state.retriever_pool_df = retriever
    if st.session_state.index:
        act = st.text_input("작업활동 및 내용 입력:")
        if st.button("유해위험요인 예측하기"):
            emb_q = embed_texts_with_openai([act], api_key=api_key)[0]
            D, I = st.session_state.index.search(np.array([emb_q], dtype='float32'), 3)
            docs = st.session_state.retriever_pool_df.iloc[I[0]]
            prompt = construct_prompt_phase1_for_hazard(docs, act)
            out = generate_with_gpt(prompt, api_key)
            hz = parse_gpt_output_phase1_for_hazard(out)
            st.write("예측 유해위험요인:", hz)
            st.write("유사 사례:")
            for r in docs.itertuples():
                st.write(f"- {r._asdict()['작업활동 및 내용']} → {r._asdict()['유해위험요인 및 환경측면 영향']}")

# Phase2 탭
with tabs[2]:
    st.markdown('<div class="sub-header">개선대책 생성 (Phase 2)</div>', unsafe_allow_html=True)
    api_key2 = st.text_input("OpenAI API 키:", type="password", key="api_key2")
    lang = st.selectbox("언어 선택:", ["Korean","English","Chinese"])
    method = st.radio("입력 방식:", ["Phase1 결과","직접 입력"])
    if method == "Phase1 결과" and 'index' in st.session_state and hasattr(st.session_state, 'retriever_pool_df'):
        # Phase1 결과 사용
        pass
    else:
        a = st.text_input("작업활동:")
        h = st.text_input("유해위험요인:")
        f = st.slider("빈도",1,5,3)
        i = st.slider("강도",1,5,3)
        T0 = f*i
        if st.button("개선대책 생성"):
            emb_q2 = embed_texts_with_openai([f"{a} {h}"], api_key=api_key2)[0]
            D2, I2 = st.session_state.index.search(np.array([emb_q2], dtype='float32'), 3)
            docs2 = st.session_state.retriever_pool_df.iloc[I2[0]]
            prompt2 = construct_prompt_phase2(docs2, a, h, f, i, T0, lang)
            out2 = generate_with_gpt(prompt2, api_key2)
            res2 = parse_gpt_output_phase2(out2)
            st.write(res2)

# 푸터
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown('<div style="display:flex;justify-content:space-between;align-items:center;">', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    if os.path.exists("cau.png"): st.image(Image.open("cau.png"), width=150)
with col2:
    if os.path.exists("doosan.png"): st.image(Image.open("doosan.png"), width=180)
st.markdown('</div>', unsafe_allow_html=True)
