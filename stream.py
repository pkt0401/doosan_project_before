import streamlit as st
import pandas as pd
import numpy as np
import faiss
import openai
import re
import os
from PIL import Image
from sklearn.model_selection import train_test_split

# 언어 설정 텍스트 정의
system_texts = {
    "Korean": {
        "title": "Artificial Intelligence Risk Assessment",
        "tab_overview": "시스템 개요",
        "tab_phase1": "위험성 평가 (Phase 1)",
        "tab_phase2": "개선대책 생성 (Phase 2)",
        "overview_header": "LLM 기반 위험성평가 시스템",
        "overview_text": """
        LLM(Large Language Model)을 활용한 위험성평가 자동화 시스템은 건설 현장의 안전 관리를 혁신적으로 개선합니다:
        
        1. <span class="highlight">작업 내용 입력 시 생성형 AI를 통한 '유해위험요인' 자동 예측 및 위험 등급 산정</span> <span class="phase-badge">Phase 1</span>
        2. <span class="highlight">위험도 감소를 위한 개선대책 자동 생성 및 감소율 예측</span> <span class="phase-badge">Phase 2</span>
        3. AI는 건설현장의 기존 위험성평가를 공정별로 구분하고, 해당 유해위험요인을 학습
        4. 자동 생성 기술 개발 완료 후 위험도 기반 사고위험성 분석 및 개선대책 생성
        
        이 시스템은 PIMS 및 안전지킴이 등 EHS 플랫폼에 AI 기술 탑재를 통해 통합 사고 예측 프로그램으로 발전 예정입니다.
        """,
        "process_title": "AI 위험성평가 프로세스",
        "process_steps": ["작업내용 입력", "AI 위험분석", "유해요인 예측", "위험등급 산정", "개선대책 자동생성", "안전조치 적용"],
        "features_title": "시스템 특징 및 구성요소",
        "phase1_features": """
        #### Phase 1: 위험성 평가 자동화
        - 공정별 작업활동에 따른 위험성평가 데이터 학습
        - 작업활동 입력 시 유해위험요인 자동 예측
        - 유사 위험요인 사례 검색 및 표시
        - 대규모 언어 모델(LLM) 기반 위험도(빈도, 강도, T) 측정
        - Excel 기반 공정별 위험성평가 데이터 자동 분석
        - 위험등급(A-E) 자동 산정
        """,
        "phase2_features": """
        #### Phase 2: 개선대책 자동 생성
        - 위험요소별 맞춤형 개선대책 자동 생성
        - 다국어(한/영/중) 개선대책 생성 지원
        - 개선 전후 위험도(T) 자동 비교 분석
        - 위험 감소율(RRR) 정량적 산출
        - 공종/공정별 최적 개선대책 데이터베이스 구축
        """,
        "phase1_header": "위험성 평가 자동화 (Phase 1)",
        "api_key_label": "OpenAI API 키를 입력하세요:",
        "dataset_label": "데이터셋 선택",
        "load_data_label": "데이터 로드 및 인덱스 구성",
        "load_data_btn": "데이터 로드 및 인덱스 구성",
        "api_key_warning": "계속하려면 OpenAI API 키를 입력하세요.",
        "data_loading": "데이터를 불러오고 인덱스를 구성하는 중...",
        "demo_limit_info": "데모 목적으로 {max_texts}개의 텍스트만 임베딩합니다. 실제 환경에서는 전체 데이터를 처리해야 합니다.",
        "data_load_success": "데이터 로드 및 인덱스 구성 완료! (총 {max_texts}개 항목 처리)",
        "hazard_prediction_header": "유해위험요인 예측",
        "load_first_warning": "먼저 [데이터 로드 및 인덱스 구성] 버튼을 클릭하세요.",
        "activity_label": "작업활동:",
        "predict_hazard_btn": "유해위험요인 예측하기",
        "activity_warning": "작업활동을 입력하세요.",
        "predicting_hazard": "유해위험요인을 예측하는 중...",
        "similar_cases_header": "유사한 사례",
        "similar_case_text": """
        <div class="similar-case">
            <strong>사례 {i}</strong><br>
            <strong>작업활동:</strong> {activity}<br>
            <strong>유해위험요인:</strong> {hazard}<br>
            <strong>위험도:</strong> 빈도 {freq}, 강도 {intensity}, T값 {t_value} (등급 {grade})
        </div>
        """,
        "prediction_result_header": "예측 결과",
        "activity_result": "작업활동: {activity}",
        "hazard_result": "예측된 유해위험요인: {hazard}",
        "result_table_columns": ["항목", "값"],
        "result_table_rows": ["빈도", "강도", "T 값", "위험등급"],
        "parsing_error": "위험성 평가 결과를 파싱할 수 없습니다.",
        "gpt_response": "GPT 원문 응답: {response}",
        "phase2_header": "개선대책 자동 생성 (Phase 2)",
        "language_select_label": "개선대책 언어 선택:",
        "input_method_label": "입력 방식 선택:",
        "input_methods": ["Phase 1 평가 결과 사용", "직접 입력"],
        "phase1_results_header": "Phase 1 평가 결과",
        "risk_level_text": "위험도: 빈도 {freq}, 강도 {intensity}, T값 {t_value} (등급 {grade})",
        "phase1_first_warning": "먼저 Phase 1에서 위험성 평가를 수행하세요.",
        "hazard_label": "유해위험요인:",
        "frequency_label": "빈도 (1-5):",
        "intensity_label": "강도 (1-5):",
        "t_value_text": "T값: {t_value} (등급: {grade})",
        "generate_improvement_btn": "개선대책 생성",
        "generating_improvement": "개선대책을 생성하는 중...",
        "no_data_warning": "Phase 1에서 데이터 로드 및 인덱스 구성을 완료하지 않았습니다. 기본 예시를 사용합니다.",
        "improvement_result_header": "개선대책 생성 결과",
        "improvement_plan_header": "개선대책",
        "risk_improvement_header": "위험도 개선 결과",
        "comparison_columns": ["항목", "개선 전", "개선 후"],
        "risk_reduction_label": "위험 감소율 (RRR)",
        "t_value_change_header": "위험도(T값) 변화",
        "before_improvement": "개선 전 T값:",
        "after_improvement": "개선 후 T값:",
        "parsing_error_improvement": "개선대책 생성 결과를 파싱할 수 없습니다."
    },
    "English": {
        "title": "Artificial Intelligence Risk Assessment",
        "tab_overview": "System Overview",
        "tab_phase1": "Risk Assessment (Phase 1)",
        "tab_phase2": "Improvement Measures (Phase 2)",
        "overview_header": "LLM-based Risk Assessment System",
        "overview_text": """
        The risk assessment automation system using LLM (Large Language Model) innovatively improves safety management at construction sites:
        
        1. <span class="highlight">Automatic prediction of 'hazards' and risk level calculation through generative AI</span> <span class="phase-badge">Phase 1</span>
        2. <span class="highlight">Automatic generation of improvement measures and reduction rate prediction to reduce risk level</span> <span class="phase-badge">Phase 2</span>
        3. AI learns existing risk assessments at construction sites by process and their hazard factors
        4. After the development of automatic generation technology, risk analysis and improvement measures based on risk level
        
        This system is expected to evolve into an integrated accident prediction program through the incorporation of AI technology into EHS platforms such as PIMS and Safety Guardian.
        """,
        "process_title": "AI Risk Assessment Process",
        "process_steps": ["Work Input", "AI Risk Analysis", "Hazard Prediction", "Risk Level Calculation", "Auto-generate Improvements", "Safety Measures"],
        "features_title": "System Features and Components",
        "phase1_features": """
        #### Phase 1: Risk Assessment Automation
        - Learning risk assessment data according to work activities by process
        - Automatic hazard prediction when work activities are entered
        - Similar case search and display
        - Risk level (frequency, intensity, T) measurement based on large language models (LLM)
        - Automatic analysis of Excel-based process-specific risk assessment data
        - Automatic risk grade (A-E) calculation
        """,
        "phase2_features": """
        #### Phase 2: Automatic Generation of Improvement Measures
        - Automatic generation of customized improvement measures for risk factors
        - Multilingual (Korean/English/Chinese) improvement measure generation support
        - Automatic comparative analysis of risk level (T) before and after improvement
        - Quantitative calculation of Risk Reduction Rate (RRR)
        - Building a database of optimal improvement measures by work type/process
        """,
        "phase1_header": "Risk Assessment Automation (Phase 1)",
        "api_key_label": "Enter OpenAI API Key:",
        "dataset_label": "Select Dataset",
        "load_data_label": "Load Data and Configure Index",
        "load_data_btn": "Load Data and Configure Index",
        "api_key_warning": "Please enter an OpenAI API key to continue.",
        "data_loading": "Loading data and configuring index...",
        "demo_limit_info": "For demo purposes, only embedding {max_texts} texts. In a real environment, all data should be processed.",
        "data_load_success": "Data load and index configuration complete! (Total {max_texts} items processed)",
        "hazard_prediction_header": "Hazard Prediction",
        "load_first_warning": "Please click the [Load Data and Configure Index] button first.",
        "activity_label": "Work Activity:",
        "predict_hazard_btn": "Predict Hazards",
        "activity_warning": "Please enter a work activity.",
        "predicting_hazard": "Predicting hazards...",
        "similar_cases_header": "Similar Cases",
        "similar_case_text": """
        <div class="similar-case">
            <strong>Case {i}</strong><br>
            <strong>Work Activity:</strong> {activity}<br>
            <strong>Hazard:</strong> {hazard}<br>
            <strong>Risk Level:</strong> Frequency {freq}, Intensity {intensity}, T-value {t_value} (Grade {grade})
        </div>
        """,
        "prediction_result_header": "Prediction Results",
        "activity_result": "Work Activity: {activity}",
        "hazard_result": "Predicted Hazard: {hazard}",
        "result_table_columns": ["Item", "Value"],
        "result_table_rows": ["Frequency", "Intensity", "T Value", "Risk Grade"],
        "parsing_error": "Unable to parse risk assessment results.",
        "gpt_response": "Original GPT Response: {response}",
        "phase2_header": "Automatic Generation of Improvement Measures (Phase 2)",
        "language_select_label": "Select Language for Improvement Measures:",
        "input_method_label": "Select Input Method:",
        "input_methods": ["Use Phase 1 Assessment Results", "Direct Input"],
        "phase1_results_header": "Phase 1 Assessment Results",
        "risk_level_text": "Risk Level: Frequency {freq}, Intensity {intensity}, T-value {t_value} (Grade {grade})",
        "phase1_first_warning": "Please perform a risk assessment in Phase 1 first.",
        "hazard_label": "Hazard:",
        "frequency_label": "Frequency (1-5):",
        "intensity_label": "Intensity (1-5):",
        "t_value_text": "T-value: {t_value} (Grade: {grade})",
        "generate_improvement_btn": "Generate Improvement Measures",
        "generating_improvement": "Generating improvement measures...",
        "no_data_warning": "Data loading and index configuration was not completed in Phase 1. Using basic examples.",
        "improvement_result_header": "Improvement Measure Generation Results",
        "improvement_plan_header": "Improvement Measures",
        "risk_improvement_header": "Risk Level Improvement Results",
        "comparison_columns": ["Item", "Before Improvement", "After Improvement"],
        "risk_reduction_label": "Risk Reduction Rate (RRR)",
        "t_value_change_header": "Risk Level (T-value) Change",
        "before_improvement": "T-value Before Improvement:",
        "after_improvement": "T-value After Improvement:",
        "parsing_error_improvement": "Unable to parse improvement measure generation results."
    },
    "Chinese": {
        "title": "Artificial Intelligence Risk Assessment",
        "tab_overview": "系统概述",
        "tab_phase1": "风险评估 (第1阶段)",
        "tab_phase2": "改进措施 (第2阶段)",
        "overview_header": "基于LLM的风险评估系统",
        "overview_text": """
        使用LLM（大型语言模型）的风险评估自动化系统创新地改善了建筑工地的安全管理：
        
        1. <span class="highlight">通过生成式AI自动预测"危害因素"并计算风险等级</span> <span class="phase-badge">第1阶段</span>
        2. <span class="highlight">自动生成改进措施并预测降低风险的比率</span> <span class="phase-badge">第2阶段</span>
        3. AI按工序学习建筑工地的现有风险评估及其危害因素
        4. 在自动生成技术开发完成后，基于风险等级进行风险分析和改进措施生成
        
        该系统有望通过将AI技术整合到EHS平台（如PIMS和安全卫士）中，发展成为一个综合事故预测程序。
        """,
        "process_title": "AI风险评估流程",
        "process_steps": ["工作输入", "AI风险分析", "危害预测", "风险等级计算", "自动生成改进措施", "安全措施"],
        "features_title": "系统特点和组件",
        "phase1_features": """
        #### 第1阶段：风险评估自动化
        - 按工序学习与工作活动相关的风险评估数据
        - 输入工作活动时自动预测危害因素
        - 相似案例搜索和显示
        - 基于大型语言模型(LLM)的风险等级（频率、强度、T值）测量
        - 自动分析基于Excel的特定工序风险评估数据
        - 自动计算风险等级(A-E)
        """,
        "phase2_features": """
        #### 第2阶段：自动生成改进措施
        - 为风险因素自动生成定制的改进措施
        - 多语言（韩语/英语/中文）改进措施生成支持
        - 改进前后风险等级（T值）的自动比较分析
        - 风险降低率(RRR)的定量计算
        - 建立按工作类型/工序的最佳改进措施数据库
        """,
        "phase1_header": "风险评估自动化 (第1阶段)",
        "api_key_label": "输入OpenAI API密钥：",
        "dataset_label": "选择数据集",
        "load_data_label": "加载数据和配置索引",
        "load_data_btn": "加载数据和配置索引",
        "api_key_warning": "请输入OpenAI API密钥以继续。",
        "data_loading": "正在加载数据和配置索引...",
        "demo_limit_info": "出于演示目的，仅嵌入{max_texts}个文本。在实际环境中，应处理所有数据。",
        "data_load_success": "数据加载和索引配置完成！（共处理{max_texts}个项目）",
        "hazard_prediction_header": "危害预测",
        "load_first_warning": "请先点击[加载数据和配置索引]按钮。",
        "activity_label": "工作活动：",
        "predict_hazard_btn": "预测危害",
        "activity_warning": "请输入工作活动。",
        "predicting_hazard": "正在预测危害...",
        "similar_cases_header": "相似案例",
        "similar_case_text": """
        <div class="similar-case">
            <strong>案例 {i}</strong><br>
            <strong>工作活动：</strong> {activity}<br>
            <strong>危害：</strong> {hazard}<br>
            <strong>风险等级：</strong> 频率 {freq}, 强度 {intensity}, T值 {t_value} (等级 {grade})
        </div>
        """,
        "prediction_result_header": "预测结果",
        "activity_result": "工作活动: {activity}",
        "hazard_result": "预测的危害: {hazard}",
        "result_table_columns": ["项目", "值"],
        "result_table_rows": ["频率", "强度", "T值", "风险等级"],
        "parsing_error": "无法解析风险评估结果。",
        "gpt_response": "原始GPT响应: {response}",
        "phase2_header": "自动生成改进措施 (第2阶段)",
        "language_select_label": "选择改进措施的语言：",
        "input_method_label": "选择输入方法：",
        "input_methods": ["使用第1阶段评估结果", "直接输入"],
        "phase1_results_header": "第1阶段评估结果",
        "risk_level_text": "风险等级: 频率 {freq}, 强度 {intensity}, T值 {t_value} (等级 {grade})",
        "phase1_first_warning": "请先在第1阶段进行风险评估。",
        "hazard_label": "危害：",
        "frequency_label": "频率 (1-5)：",
        "intensity_label": "强度 (1-5)：",
        "t_value_text": "T值: {t_value} (等级: {grade})",
        "generate_improvement_btn": "生成改进措施",
        "generating_improvement": "正在生成改进措施...",
        "no_data_warning": "在第1阶段未完成数据加载和索引配置。使用基本示例。",
        "improvement_result_header": "改进措施生成结果",
        "improvement_plan_header": "改进措施",
        "risk_improvement_header": "风险等级改进结果",
        "comparison_columns": ["项目", "改进前", "改进后"],
        "risk_reduction_label": "风险降低率 (RRR)",
        "t_value_change_header": "风险等级 (T值) 变化",
        "before_improvement": "改进前T值：",
        "after_improvement": "改进后T值：",
        "parsing_error_improvement": "无法解析改进措施生成结果。"
    }
}

# 페이지 설정
st.set_page_config(
    page_title="Artificial Intelligence Risk Assessment",
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
    .similar-case {
        background-color: #f1f8e9;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        border-left: 4px solid #689f38;
    }
    .language-selector {
        position: absolute;
        top: 10px;
        right: 10px;
        z-index: 1000;
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if "index" not in st.session_state:
    st.session_state.index = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "retriever_pool_df" not in st.session_state:
    st.session_state.retriever_pool_df = None
if "language" not in st.session_state:
    st.session_state.language = "Korean"

# 상단에 언어 선택기 추가
col1, col2 = st.columns([6, 1])
with col2:
    selected_language = st.selectbox(
        "",
        options=list(system_texts.keys()),
        index=list(system_texts.keys()).index(st.session_state.language) if st.session_state.language in system_texts else 0,
        key="language_selector"
    )
    st.session_state.language = selected_language

# 현재 언어에 따른 텍스트 가져오기
texts = system_texts[st.session_state.language]

# 헤더 표시
st.markdown(f'<div class="main-header">{texts["title"]}</div>', unsafe_allow_html=True)

# 탭 설정
tabs = st.tabs([texts["tab_overview"], texts["tab_phase1"], texts["tab_phase2"]])

# ------------------ 유틸리티 함수 ------------------

# 빈도*강도 결과 T에 따른 등급 결정 함수
def determine_grade(value):
    """빈도*강도 결과 T에 따른 등급 결정 함수."""
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
        return '알 수 없음' if st.session_state.language == 'Korean' else 'Unknown'

# 데이터 불러오기 함수
def load_data(selected_dataset_name):
    """선택된 이름에 대응하는 Excel 데이터 불러오기."""
    try:
        # 실제 Excel 파일에서 데이터 로드
        df = pd.read_excel(f"{selected_dataset_name}.xlsx")

        # 전처리
        if '삭제 Del' in df.columns:
            df = df.drop(['삭제 Del'], axis=1)
        df = df.iloc[1:]
        df = df.rename(columns={df.columns[4]: '빈도'})
        df = df.rename(columns={df.columns[5]: '강도'})

        df['T'] = pd.to_numeric(df.iloc[:, 4]) * pd.to_numeric(df.iloc[:, 5])
        df = df.iloc[:, :7]
        df.rename(
            columns={
                '작업활동 및 내용\nWork & Contents': '작업활동 및 내용',
                '유해위험요인 및 환경측면 영향\nHazard & Risk': '유해위험요인 및 환경측면 영향',
                '피해형태 및 환경영향\nDamage & Effect': '피해형태 및 환경영향'
            },
            inplace=True
        )
        df = df.rename(columns={df.columns[6]: 'T'})
        df['등급'] = df['T'].apply(determine_grade)

        return df
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {str(e)}")
        st.write(f"시도한 파일 경로: {selected_dataset_name}")
        
        # 파일이 없는 경우 대비한 더미 데이터 생성 (데모용)
        st.warning("Excel 파일을 찾을 수 없어 샘플 데이터를 생성합니다. 실제 운영 시에는 해당 파일이 필요합니다.")
        data = {
            "작업활동 및 내용": ["Shoring Installation", "In and Out of materials", "Transport / Delivery", "Survey and Inspection"],
            "유해위험요인 및 환경측면 영향": ["Fall and collision due to unstable ground", "Overturning of transport vehicle", 
                                 "Collision between transport vehicle", "Personnel fall while inspecting"],
            "피해형태 및 환경영향": ["Injury", "Equipment damage", "Collision injury", "Fall injury"],
            "빈도": [3, 3, 3, 2],
            "강도": [2, 3, 5, 3]
        }
        
        df = pd.DataFrame(data)
        df['T'] = df['빈도'] * df['강도']
        df['등급'] = df['T'].apply(determine_grade)
        
        return df

# OpenAI 임베딩 API를 통해 텍스트 임베딩 생성
def embed_texts_with_openai(texts, model="text-embedding-3-large", api_key=None):
    """OpenAI 임베딩 API로 텍스트 리스트를 임베딩."""
    if api_key:
        openai.api_key = api_key
    
    embeddings = []
    progress_bar = st.progress(0)
    total = len(texts)

    for idx, text in enumerate(texts):
        try:
            text = str(text).replace("\n", " ")
            response = openai.Embedding.create(model=model, input=[text])
            embedding = response["data"][0]["embedding"]
            embeddings.append(embedding)
        except Exception as e:
            st.error(f"텍스트 임베딩 중 오류 발생: {str(e)}")
            embeddings.append([0]*1536)
        
        progress_bar.progress((idx + 1) / total)
    
    return embeddings

# GPT 모델을 통해 예측 결과 생성
def generate_with_gpt(prompt, api_key=None, model="gpt-4o", language="Korean"):
    """GPT 모델로부터 예측 결과를 받아오는 함수."""
    if api_key:
        openai.api_key = api_key
        
    # 언어에 따른 시스템 프롬프트 설정
    system_prompts = {
        "Korean": "위험성 평가 및 개선대책 생성을 돕는 도우미입니다. 한국어로 응답하세요.",
        "English": "I am an assistant helping with risk assessment and improvement measures. Please respond in English.",
        "Chinese": "我是一个协助进行风险评估和改进措施的助手。请用中文回答。"
    }
    
    system_prompt = system_prompts.get(language, system_prompts["Korean"])
        
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=250
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        st.error(f"GPT API 호출 중 오류 발생: {str(e)}")
        return None

# ----- Phase 1: 유해위험요인 예측 관련 함수 -----

# 검색된 문서로 GPT 프롬프트 생성 (Phase 1 - 유해위험요인 예측)
def construct_prompt_phase1_hazard(retrieved_docs, activity_text, language="Korean"):
    """작업활동으로부터 유해위험요인을 예측하는 프롬프트 생성."""
    # 언어에 따른 프롬프트 템플릿
    prompt_templates = {
        "Korean": {
            "intro": "다음은 건설 현장의 작업활동과 그에 따른 유해위험요인의 예시입니다:\n\n",
            "example_format": "예시 {i}:\n작업활동: {activity}\n유해위험요인: {hazard}\n\n",
            "query_format": "이제 다음 작업활동에 대한 유해위험요인을 예측해주세요:\n작업활동: {activity}\n유해위험요인: "
        },
        "English": {
            "intro": "The following are examples of work activities at construction sites and their associated hazards:\n\n",
            "example_format": "Example {i}:\nWork Activity: {activity}\nHazard: {hazard}\n\n",
            "query_format": "Now, please predict the hazard for the following work activity:\nWork Activity: {activity}\nHazard: "
        },
        "Chinese": {
            "intro": "以下是建筑工地的工作活动及其相关危害的例子:\n\n",
            "example_format": "例子 {i}:\n工作活动: {activity}\n危害: {hazard}\n\n",
            "query_format": "现在，请预测以下工作活动的危害:\n工作活动: {activity}\n危害: "
        }
    }
    
    # 현재 언어의 템플릿 가져오기
    template = prompt_templates.get(language, prompt_templates["Korean"])
    
    retrieved_examples = []
    for _, doc in retrieved_docs.iterrows():
        try:
            activity = doc['작업활동 및 내용']
            hazard = doc['유해위험요인 및 환경측면 영향']
            retrieved_examples.append((activity, hazard))
        except:
            continue
    
    prompt = template["intro"]
    for i, (activity, hazard) in enumerate(retrieved_examples, 1):
        prompt += template["example_format"].format(i=i, activity=activity, hazard=hazard)
    
    prompt += template["query_format"].format(activity=activity_text)
    
    return prompt

# 빈도와 강도 예측을 위한 프롬프트 생성 (Phase 1)
def construct_prompt_phase1_risk(retrieved_docs, activity_text, hazard_text, language="Korean"):
    """작업활동과 유해위험요인을 바탕으로 빈도와 강도를 예측하는 프롬프트 생성."""
    # 언어에 따른 프롬프트 템플릿
    prompt_templates = {
        "Korean": {
            "example_format": "예시 {i}:\n입력: {input}\n출력: {output}\n\n",
            "query_format": "입력: {activity} - {hazard}\n위 입력을 바탕으로 빈도와 강도를 예측하세요. 빈도는 1에서 5 사이의 정수입니다. 강도는 1에서 5 사이의 정수입니다. T는 빈도와 강도를 곱한 값입니다.\n다음 JSON 형식으로 출력하세요:\n{json_format}\n출력:\n"
        },
        "English": {
            "example_format": "Example {i}:\nInput: {input}\nOutput: {output}\n\n",
            "query_format": "Input: {activity} - {hazard}\nBased on the above input, predict frequency and intensity. Frequency is an integer between 1 and 5. Intensity is an integer between 1 and 5. T is the product of frequency and intensity.\nOutput in the following JSON format:\n{json_format}\nOutput:\n"
        },
        "Chinese": {
            "example_format": "示例 {i}:\n输入: {input}\n输出: {output}\n\n",
            "query_format": "输入: {activity} - {hazard}\n根据上述输入，预测频率和强度。频率是1到5之间的整数。强度是1到5之间的整数。T是频率和强度的乘积。\n请以下列JSON格式输出:\n{json_format}\n输出:\n"
        }
    }
    
    # JSON 형식 언어별 정의
    json_formats = {
        "Korean": '{"빈도": 숫자, "강도": 숫자, "T": 숫자}',
        "English": '{"frequency": number, "intensity": number, "T": number}',
        "Chinese": '{"频率": 数字, "强度": 数字, "T": 数字}'
    }
    
    # 현재 언어의 템플릿 가져오기
    template = prompt_templates.get(language, prompt_templates["Korean"])
    json_format = json_formats.get(language, json_formats["Korean"])
    
    retrieved_examples = []
    for _, doc in retrieved_docs.iterrows():
        try:
            example_input = f"{doc['작업활동 및 내용']} - {doc['유해위험요인 및 환경측면 영향']}"
            frequency = int(doc['빈도'])
            intensity = int(doc['강도'])
            T_value = frequency * intensity
            
            # 언어별 JSON 출력 형식
            if language == "Korean":
                example_output = f'{{"빈도": {frequency}, "강도": {intensity}, "T": {T_value}}}'
            elif language == "English":
                example_output = f'{{"frequency": {frequency}, "intensity": {intensity}, "T": {T_value}}}'
            elif language == "Chinese":
                example_output = f'{{"频率": {frequency}, "强度": {intensity}, "T": {T_value}}}'
            else:
                example_output = f'{{"빈도": {frequency}, "강도": {intensity}, "T": {T_value}}}'
                
            retrieved_examples.append((example_input, example_output))
        except:
            continue
    
    prompt = ""
    for i, (example_input, example_output) in enumerate(retrieved_examples, 1):
        prompt += template["example_format"].format(i=i, input=example_input, output=example_output)
    
    prompt += template["query_format"].format(
        activity=activity_text, 
        hazard=hazard_text,
        json_format=json_format
    )
    
    return prompt

# GPT 출력 파싱 (Phase 1)
def parse_gpt_output_phase1(gpt_output, language="Korean"):
    """GPT 출력에서 {빈도, 강도, T}를 정규표현식으로 추출."""
    # 언어별 JSON 패턴
    json_patterns = {
        "Korean": r'\{"빈도":\s*([1-5]),\s*"강도":\s*([1-5]),\s*"T":\s*([0-9]+)\}',
        "English": r'\{"frequency":\s*([1-5]),\s*"intensity":\s*([1-5]),\s*"T":\s*([0-9]+)\}',
        "Chinese": r'\{"频率":\s*([1-5]),\s*"强度":\s*([1-5]),\s*"T":\s*([0-9]+)\}'
    }
    
    pattern = json_patterns.get(language, json_patterns["Korean"])
    match = re.search(pattern, gpt_output)
    
    if match:
        pred_frequency = int(match.group(1))
        pred_intensity = int(match.group(2))
        pred_T = int(match.group(3))
        return pred_frequency, pred_intensity, pred_T
    else:
        # 다른 패턴도 시도 (GPT가 다른 언어로 응답했을 경우)
        for lang, pattern in json_patterns.items():
            if lang != language:  # 이미 시도한 언어는 건너뛴다
                match = re.search(pattern, gpt_output)
                if match:
                    pred_frequency = int(match.group(1))
                    pred_intensity = int(match.group(2))
                    pred_T = int(match.group(3))
                    return pred_frequency, pred_intensity, pred_T
        
        return None

# ----- Phase 2: 개선대책 생성 관련 함수 -----

# 위험 감소율(RRR) 계산 함수
def compute_rrr(T_before, T_after):
    """위험 감소율(Risk Reduction Rate) 계산"""
    if T_before == 0:
        return 0.0
    return ((T_before - T_after) / T_before) * 100.0

# 개선대책 생성을 위한 프롬프트 구성 (Phase 2)
def construct_prompt_phase2(retrieved_docs, activity_text, hazard_text, freq, intensity, T, target_language="Korean"):
    """
    개선대책 생성을 위한 프롬프트 구성
    """
    # 예시 섹션 구성
    example_section = ""
    examples_added = 0
    
    # 언어별 필드명
    field_names = {
        "Korean": {
            "improvement_fields": ['개선대책 및 세부관리방안', '개선대책', '개선방안', 'Corrective Action'],
            "activity": "작업활동 및 내용",
            "hazard": "유해위험요인 및 환경측면 영향",
            "freq": "빈도",
            "intensity": "강도",
            "example_intro": "Example:",
            "input_activity": "Input (Activity): ",
            "input_hazard": "Input (Hazard): ",
            "input_freq": "Input (Original Frequency): ",
            "input_intensity": "Input (Original Intensity): ",
            "input_t": "Input (Original T): ",
            "output_intro": "Output (Improvement Plan and Risk Reduction) in JSON:",
            "improvement": "개선대책",
            "improved_freq": "개선 후 빈도",
            "improved_intensity": "개선 후 강도",
            "improved_t": "개선 후 T",
            "reduction_rate": "T 감소율"
        },
        "English": {
            "improvement_fields": ['Improvement Measures', 'Improvement Plan', 'Countermeasures'],
            "activity": "작업활동 및 내용",
            "hazard": "유해위험요인 및 환경측면 영향",
            "freq": "빈도",
            "intensity": "강도",
            "example_intro": "Example:",
            "input_activity": "Input (Activity): ",
            "input_hazard": "Input (Hazard): ",
            "input_freq": "Input (Original Frequency): ",
            "input_intensity": "Input (Original Intensity): ",
            "input_t": "Input (Original T): ",
            "output_intro": "Output (Improvement Plan and Risk Reduction) in JSON:",
            "improvement": "improvement_plan",
            "improved_freq": "improved_frequency",
            "improved_intensity": "improved_intensity",
            "improved_t": "improved_T",
            "reduction_rate": "reduction_rate"
        },
        "Chinese": {
            "improvement_fields": ['改进措施', '改进计划', '对策'],
            "activity": "작업활동 및 내용",
            "hazard": "유해위험요인 및 환경측면 영향",
            "freq": "빈도",
            "intensity": "강도",
            "example_intro": "示例:",
            "input_activity": "输入 (工作活动): ",
            "input_hazard": "输入 (危害): ",
            "input_freq": "输入 (原频率): ",
            "input_intensity": "输入 (原强度): ",
            "input_t": "输入 (原T值): ",
            "output_intro": "输出 (改进计划和风险降低) 以JSON格式:",
            "improvement": "改进措施",
            "improved_freq": "改进后频率",
            "improved_intensity": "改进后强度",
            "improved_t": "改进后T值",
            "reduction_rate": "T值降低率"
        }
    }
    
    # 현재 언어에 맞는 필드명 가져오기
    fields = field_names.get(target_language, field_names["Korean"])
    
    for _, row in retrieved_docs.iterrows():
        try:
            # Phase2 데이터셋에 있는 개선대책 필드 사용 시도
            improvement_plan = ""
            for field in fields["improvement_fields"]:
                if field in row and pd.notna(row[field]):
                    improvement_plan = row[field]
                    break
            
            if not improvement_plan:
                continue  # 개선대책이 없으면 건너뛰기
                
            original_freq = int(row[fields["freq"]]) if fields["freq"] in row else 3
            original_intensity = int(row[fields["intensity"]]) if fields["intensity"] in row else 3
            original_T = original_freq * original_intensity
                
            # 개선 후 데이터 시도
            improved_freq = 1
            improved_intensity = 1
            improved_T = 1
            
            for field_pattern in [('개선 후 빈도', '개선 후 강도', '개선 후 T'), ('개선빈도', '개선강도', '개선T')]:
                if all(field in row for field in field_pattern):
                    improved_freq = int(row[field_pattern[0]])
                    improved_intensity = int(row[field_pattern[1]])
                    improved_T = int(row[field_pattern[2]])
                    break
            
            example_section += (
                f"{fields['example_intro']}\n"
                f"{fields['input_activity']}{row[fields['activity']]}\n"
                f"{fields['input_hazard']}{row[fields['hazard']]}\n"
                f"{fields['input_freq']}{original_freq}\n"
                f"{fields['input_intensity']}{original_intensity}\n"
                f"{fields['input_t']}{original_T}\n"
                f"{fields['output_intro']}\n"
                "{\n"
                f'  "{fields["improvement"]}": "{improvement_plan}",\n'
                f'  "{fields["improved_freq"]}": {improved_freq},\n'
                f'  "{fields["improved_intensity"]}": {improved_intensity},\n'
                f'  "{fields["improved_t"]}": {improved_T},\n'
                f'  "{fields["reduction_rate"]}": {compute_rrr(original_T, improved_T):.2f}\n'
                "}\n\n"
            )
            
            examples_added += 1
            if examples_added >= 3:  # 최대 3개 예시만 사용
                break
                
        except Exception as e:
            # 에러 발생 시 해당 예시 건너뛰기
            continue
    
    # 예시가 없는 경우 기본 예시 추가 (언어별)
    if examples_added == 0:
        # 한국어 기본 예시
        if target_language == "Korean":
            example_section = """
Example:
Input (Activity): Excavation and backfilling
Input (Hazard): Collapse of excavation wall due to improper sloping
Input (Original Frequency): 3
Input (Original Intensity): 4
Input (Original T): 12
Output (Improvement Plan and Risk Reduction) in JSON:
{
  "개선대책": "1) 토양 분류에 따른 적절한 경사 유지 2) 굴착 벽면 보강 3) 정기적인 지반 상태 검사 실시",
  "개선 후 빈도": 1,
  "개선 후 강도": 2,
  "개선 후 T": 2,
  "T 감소율": 83.33
}

Example:
Input (Activity): Lifting operation
Input (Hazard): Material fall due to improper rigging
Input (Original Frequency): 2
Input (Original Intensity): 5
Input (Original T): 10
Output (Improvement Plan and Risk Reduction) in JSON:
{
  "개선대책": "1) 리깅 전문가 작업 참여 2) 리깅 장비 사전 점검 3) 안전 구역 설정 및 접근 통제",
  "개선 후 빈도": 1,
  "개선 후 강도": 2,
  "개선 후 T": 2,
  "T 감소율": 80.00
}
"""
        # 영어 기본 예시
        elif target_language == "English":
            example_section = """
Example:
Input (Activity): Excavation and backfilling
Input (Hazard): Collapse of excavation wall due to improper sloping
Input (Original Frequency): 3
Input (Original Intensity): 4
Input (Original T): 12
Output (Improvement Plan and Risk Reduction) in JSON:
{
  "improvement_plan": "1) Maintain proper slope according to soil classification 2) Reinforce excavation walls 3) Conduct regular ground condition inspections",
  "improved_frequency": 1,
  "improved_intensity": 2,
  "improved_T": 2,
  "reduction_rate": 83.33
}

Example:
Input (Activity): Lifting operation
Input (Hazard): Material fall due to improper rigging
Input (Original Frequency): 2
Input (Original Intensity): 5
Input (Original T): 10
Output (Improvement Plan and Risk Reduction) in JSON:
{
  "improvement_plan": "1) Involve rigging experts in operations 2) Pre-inspect rigging equipment 3) Set up safety zones and control access",
  "improved_frequency": 1,
  "improved_intensity": 2,
  "improved_T": 2,
  "reduction_rate": 80.00
}
"""
        # 중국어 기본 예시
        elif target_language == "Chinese":
            example_section = """
示例:
输入 (工作活动): Excavation and backfilling
输入 (危害): Collapse of excavation wall due to improper sloping
输入 (原频率): 3
输入 (原强度): 4
输入 (原T值): 12
输出 (改进计划和风险降低) 以JSON格式:
{
  "改进措施": "1) 根据土壤分类维持适当的斜坡 2) 加固挖掘墙壁 3) 定期进行地面状况检查",
  "改进后频率": 1,
  "改进后强度": 2,
  "改进后T值": 2,
  "T值降低率": 83.33
}

示例:
输入 (工作活动): Lifting operation
输入 (危害): Material fall due to improper rigging
输入 (原频率): 2
输入 (原强度): 5
输入 (原T值): 10
输出 (改进计划和风险降低) 以JSON格式:
{
  "改进措施": "1) 吊装专家参与作业 2) 预检查吊装设备 3) 设置安全区域并控制进入",
  "改进后频率": 1,
  "改进后强度": 2,
  "改进后T值": 2,
  "T值降低率": 80.00
}
"""
    
    # 언어별 JSON 출력 키 이름
    json_keys = {
        "Korean": {
            "improvement": "개선대책",
            "improved_freq": "개선 후 빈도",
            "improved_intensity": "개선 후 강도",
            "improved_t": "개선 후 T",
            "reduction_rate": "T 감소율"
        },
        "English": {
            "improvement": "improvement_plan",
            "improved_freq": "improved_frequency",
            "improved_intensity": "improved_intensity",
            "improved_t": "improved_T",
            "reduction_rate": "reduction_rate"
        },
        "Chinese": {
            "improvement": "改进措施",
            "improved_freq": "改进后频率",
            "improved_intensity": "改进后强度", 
            "improved_t": "改进后T值",
            "reduction_rate": "T值降低率"
        }
    }
    
    # 각 언어별 안내 메시지
    instructions = {
        "Korean": {
            "new_input": "다음은 새로운 입력입니다:",
            "input_activity": "입력 (작업활동): ",
            "input_hazard": "입력 (유해위험요인): ",
            "input_freq": "입력 (원래 빈도): ",
            "input_intensity": "입력 (원래 강도): ",
            "input_t": "입력 (원래 T): ",
            "output_format": "다음 JSON 형식으로 출력을 제공하세요:",
            "improvement_write": "개선대책(개선대책)은 한국어로 작성하세요.",
            "provide_measures": "최소 3개의 구체적인 개선 조치를 번호가 매겨진 목록으로 제공하세요.",
            "valid_json": "유효한 JSON만 반환하도록 하세요.",
            "output": "출력:"
        },
        "English": {
            "new_input": "Now here is a new input:",
            "input_activity": "Input (Activity): ",
            "input_hazard": "Input (Hazard): ",
            "input_freq": "Input (Original Frequency): ",
            "input_intensity": "Input (Original Intensity): ",
            "input_t": "Input (Original T): ",
            "output_format": "Please provide the output in JSON format with these keys:",
            "improvement_write": "Please write the improvement measures (improvement_plan) in English.",
            "provide_measures": "Provide at least 3 specific improvement measures as a numbered list.",
            "valid_json": "Make sure to return only valid JSON.",
            "output": "Output:"
        },
        "Chinese": {
            "new_input": "以下是新的输入:",
            "input_activity": "输入 (工作活动): ",
            "input_hazard": "输入 (危害): ",
            "input_freq": "输入 (原频率): ",
            "input_intensity": "输入 (原强度): ",
            "input_t": "输入 (原T值): ",
            "output_format": "请以以下JSON格式提供输出:",
            "improvement_write": "请用中文编写改进措施(改进措施)。",
            "provide_measures": "提供至少3项具体的改进措施，列为编号列表。",
            "valid_json": "请确保只返回有效的JSON。",
            "output": "输出:"
        }
    }
    
    # 현재 언어의 키와 안내 메시지
    keys = json_keys.get(target_language, json_keys["Korean"])
    instr = instructions.get(target_language, instructions["Korean"])
    
    # 최종 프롬프트 생성
    prompt = (
        f"{example_section}"
        f"{instr['new_input']}\n"
        f"{instr['input_activity']}{activity_text}\n"
        f"{instr['input_hazard']}{hazard_text}\n"
        f"{instr['input_freq']}{freq}\n"
        f"{instr['input_intensity']}{intensity}\n"
        f"{instr['input_t']}{T}\n\n"
        f"{instr['output_format']}\n"
        "{\n"
        f'  "{keys["improvement"]}": "항목별 개선대책 리스트", \n'
        f'  "{keys["improved_freq"]}": (an integer in [1..5]),\n'
        f'  "{keys["improved_intensity"]}": (an integer in [1..5]),\n'
        f'  "{keys["improved_t"]}": (Improved Frequency * Improved Severity),\n'
        f'  "{keys["reduction_rate"]}": (percentage of risk reduction)\n'
        "}\n\n"
        f"{instr['improvement_write']}\n"
        f"{instr['provide_measures']}\n"
        f"{instr['valid_json']}\n"
        f"{instr['output']}\n"
    )
    
    return prompt

# GPT 응답 파싱 (Phase 2)
def parse_gpt_output_phase2(gpt_output, language="Korean"):
    """GPT 응답에서 JSON 데이터를 추출"""
    try:
        # JSON 블록이 있는 경우 추출 시도
        pattern = re.compile(r"```json(.*?)```", re.DOTALL)
        match = pattern.search(gpt_output)

        if match:
            json_str = match.group(1).strip()
        else:
            # JSON 블록 표시가 없는 경우 원문을 JSON으로 파싱 시도
            json_str = gpt_output.replace("```", "").strip()

        import json
        result = json.loads(json_str)
        
        # 언어별 키 매핑
        key_mappings = {
            "Korean": {
                "improvement": ["개선대책"],
                "improved_freq": ["개선 후 빈도", "개선빈도"],
                "improved_intensity": ["개선 후 강도", "개선강도"],
                "improved_t": ["개선 후 T", "개선T", "개선 후 t"],
                "reduction_rate": ["T 감소율", "감소율", "위험 감소율"]
            },
            "English": {
                "improvement": ["improvement_plan", "improvement_measures", "improvements"],
                "improved_freq": ["improved_frequency", "new_frequency", "frequency_after"],
                "improved_intensity": ["improved_intensity", "new_intensity", "intensity_after"],
                "improved_t": ["improved_T", "new_T", "T_after"],
                "reduction_rate": ["reduction_rate", "risk_reduction_rate", "rrr"]
            },
            "Chinese": {
                "improvement": ["改进措施", "改进计划", "改善措施"],
                "improved_freq": ["改进后频率", "新频率", "频率改进后"],
                "improved_intensity": ["改进后强度", "新强度", "强度改进后"],
                "improved_t": ["改进后T值", "新T值", "T值改进后"],
                "reduction_rate": ["T值降低率", "风险降低率", "降低率"]
            }
        }
        
        # 결과 매핑
        mapped_result = {}
        
        # 현재 언어의 키 매핑 가져오기
        mappings = key_mappings.get(language, key_mappings["Korean"])
        
        # 개선대책 키 매핑
        for result_key, possible_keys in mappings.items():
            for key in possible_keys:
                if key in result:
                    mapped_result[result_key] = result[key]
                    break
            
        return mapped_result
    except Exception as e:
        st.error(f"JSON 파싱 중 오류 발생: {str(e)}")
        st.write("원본 GPT 응답:", gpt_output)
        return None

# ------------------ 데이터셋 및 샘플 데이터 준비 ------------------

# 데이터셋 옵션
dataset_options = {
    "SWRO 건축공정 (건축)": "SWRO 건축공정 (건축)",
    "Civil (토목)": "Civil (토목)",
    "Marine (토목)": "Marine (토목)",
    "SWRO 기계공사 (플랜트)": "SWRO 기계공사 (플랜트)",
    "SWRO 전기작업표준 (플랜트)": "SWRO 전기작업표준 (플랜트)"
}

# ----- 시스템 개요 탭 -----
with tabs[0]:
    st.markdown(f'<div class="sub-header">{texts["overview_header"]}</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown(f"""
        <div class="info-text">
        {texts["overview_text"]}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # AI 위험성평가 프로세스 다이어그램
        st.markdown(f'<div style="text-align: center; margin-bottom: 10px;"><b>{texts["process_title"]}</b></div>', unsafe_allow_html=True)
        
        steps = texts["process_steps"]
        
        for i, step in enumerate(steps):
            phase_badge = '<span class="phase-badge">Phase 1</span>' if i < 4 else '<span class="phase-badge">Phase 2</span>'
            st.markdown(f"**{i+1}. {step}** {phase_badge}" + (" → " if i < len(steps)-1 else ""), unsafe_allow_html=True)
    
    # 시스템 특징
    st.markdown(f'<div class="sub-header">{texts["features_title"]}</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(texts["phase1_features"], unsafe_allow_html=True)
    
    with col2:
        st.markdown(texts["phase2_features"], unsafe_allow_html=True)

# ----- Phase 1: 위험성 평가 탭 -----
with tabs[1]:
    st.markdown(f'<div class="sub-header">{texts["phase1_header"]}</div>', unsafe_allow_html=True)
    
    # API 키 입력
    api_key = st.text_input(texts["api_key_label"], type="password", key="api_key_phase1")
    
    # 데이터셋 선택
    selected_dataset_name = st.selectbox(
        texts["dataset_label"],
        options=list(dataset_options.keys()),
        key="dataset_selector_phase1"
    )
    
    # 인덱스 구성 섹션
    st.markdown(f"### {texts['load_data_label']}")
    
    if st.button(texts["load_data_btn"], key="load_data_phase1"):
        if not api_key:
            st.warning(texts["api_key_warning"])
        else:
            with st.spinner(texts["data_loading"]):
                # 데이터 불러오기
                df = load_data(dataset_options[selected_dataset_name])
                
                if df is not None:
                    # Train/Test 분할
                    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
                    
                    # 리트리버 풀 구성
                    retriever_pool_df = train_df.copy()
                    retriever_pool_df['content'] = retriever_pool_df.apply(
                        lambda row: ' '.join(row.values.astype(str)), axis=1
                    )
                    texts_to_embed = retriever_pool_df['content'].tolist()
                    
                    # 임베딩 생성 (데모에서는 적은 수만 처리, 실제로는 전체 처리)
                    max_texts = min(len(texts_to_embed), 10)  # 데모에서는 최대 10개만 처리
                    st.info("텍스트 임베딩 중입니다...")
                    
                    openai.api_key = api_key
                    embeddings = embed_texts_with_openai(texts_to_embed[:max_texts], api_key=api_key)
                    
                    # FAISS 인덱스 구성
                    embeddings_array = np.array(embeddings, dtype='float32')
                    dimension = embeddings_array.shape[1]
                    faiss_index = faiss.IndexFlatL2(dimension)
                    faiss_index.add(embeddings_array)
                    
                    # 세션 상태에 저장
                    st.session_state.index = faiss_index
                    st.session_state.embeddings = embeddings_array
                    st.session_state.retriever_pool_df = retriever_pool_df.iloc[:max_texts]  # 임베딩된 부분만 저장
                    
                    st.success("데이터 로드 및 인덱스 구성 완료!")
                    st.session_state.test_df = test_df
    
    # 사용자 입력 예측 섹션
    st.markdown(f"### {texts['hazard_prediction_header']}")
    
    if st.session_state.index is None:
        st.warning(texts["load_first_warning"])
    else:
        with st.form("user_input_form"):
            user_work = st.text_input(texts["activity_label"], key="form_user_work")
            submitted = st.form_submit_button(texts["predict_hazard_btn"])
            
        if submitted:
            if not user_work:
                st.warning(texts["activity_warning"])
            else:
                with st.spinner(texts["predicting_hazard"]):
                    # 쿼리 임베딩
                    query_embedding = embed_texts_with_openai([user_work], api_key=api_key)[0]
                    query_embedding_array = np.array([query_embedding], dtype='float32')
                    
                    # 유사 문서 검색
                    k_similar = min(3, len(st.session_state.retriever_pool_df))
                    distances, indices = st.session_state.index.search(query_embedding_array, k_similar)
                    retrieved_docs = st.session_state.retriever_pool_df.iloc[indices[0]]
                    
                    # 유사한 사례 표시
                    st.markdown(f"#### {texts['similar_cases_header']}")
                    for i, (_, doc) in enumerate(retrieved_docs.iterrows(), 1):
                        st.markdown(
                            texts["similar_case_text"].format(
                                i=i,
                                activity=doc['작업활동 및 내용'],
                                hazard=doc['유해위험요인 및 환경측면 영향'],
                                freq=doc['빈도'],
                                intensity=doc['강도'],
                                t_value=doc['T'],
                                grade=doc['등급']
                            ), 
                            unsafe_allow_html=True
                        )
                    
                    # GPT 프롬프트 생성 & 호출 (유해위험요인 예측)
                    hazard_prompt = construct_prompt_phase1_hazard(retrieved_docs, user_work, language=st.session_state.language)
                    hazard_prediction = generate_with_gpt(hazard_prompt, api_key=api_key, language=st.session_state.language)
                    
                    # 빈도와 강도 예측을 위한 프롬프트 생성 & 호출
                    risk_prompt = construct_prompt_phase1_risk(retrieved_docs, user_work, hazard_prediction, language=st.session_state.language)
                    risk_prediction = generate_with_gpt(risk_prompt, api_key=api_key, language=st.session_state.language)
                    
                    # 결과 표시
                    st.markdown(f"#### {texts['prediction_result_header']}")
                    st.markdown(texts["activity_result"].format(activity=user_work))
                    st.markdown(texts["hazard_result"].format(hazard=hazard_prediction))
                    
                    parse_result = parse_gpt_output_phase1(risk_prediction, language=st.session_state.language)
                    if parse_result is not None:
                        f_val, i_val, t_val = parse_result
                        grade = determine_grade(t_val)
                        
                        # 결과를 표로 표시
                        result_df = pd.DataFrame({
                            texts["result_table_columns"][0]: texts["result_table_rows"],
                            texts["result_table_columns"][1]: [f_val, i_val, t_val, grade]
                        })
                        
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.table(result_df)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # 세션 상태에 결과 저장 (Phase 2에서 사용)
                        st.session_state.last_assessment = {
                            'activity': user_work,
                            'hazard': hazard_prediction,
                            'frequency': f_val,
                            'intensity': i_val,
                            'T': t_val,
                            'grade': grade
                        }
                    else:
                        st.error(texts["parsing_error"])
                        st.write(texts["gpt_response"].format(response=risk_prediction))

# ----- Phase 2: 개선대책 생성 탭 -----
with tabs[2]:
    st.markdown(f'<div class="sub-header">{texts["phase2_header"]}</div>', unsafe_allow_html=True)
    
    # API 키 입력
    api_key_phase2 = st.text_input(texts["api_key_label"], type="password", key="api_key_phase2")
    
    # 개선대책 언어 선택
    target_language = st.selectbox(
        texts["language_select_label"],
        options=list(system_texts.keys()),
        index=list(system_texts.keys()).index(st.session_state.language),
        key="target_language"
    )
    
    # 입력 방식 선택 (Phase 1 결과 사용 또는 직접 입력)
    input_method = st.radio(
        texts["input_method_label"],
        options=texts["input_methods"],
        index=0,
        key="input_method"
    )
    
    if input_method == texts["input_methods"][0]:  # Phase 1 평가 결과 사용
        # Phase 1 결과를 사용하는 경우
        if hasattr(st.session_state, 'last_assessment'):
            last_assessment = st.session_state.last_assessment
            
            st.markdown(f"### {texts['phase1_results_header']}")
            st.markdown(f"**{texts['activity_label'].strip(':')}** {last_assessment['activity']}")
            st.markdown(f"**{texts['hazard_label'].strip(':')}** {last_assessment['hazard']}")
            st.markdown(
                texts["risk_level_text"].format(
                    freq=last_assessment['frequency'],
                    intensity=last_assessment['intensity'],
                    t_value=last_assessment['T'],
                    grade=last_assessment['grade']
                )
            )
            
            activity_text = last_assessment['activity']
            hazard_text = last_assessment['hazard']
            frequency = last_assessment['frequency']
            intensity = last_assessment['intensity']
            T_value = last_assessment['T']
            
        else:
            st.warning(texts["phase1_first_warning"])
            activity_text = hazard_text = None
            frequency = intensity = T_value = None
    else:
        # 직접 입력하는 경우
        col1, col2 = st.columns(2)
        
        with col1:
            activity_text = st.text_input(texts["activity_label"], key="direct_activity")
            hazard_text = st.text_input(texts["hazard_label"], key="direct_hazard")
        
        with col2:
            frequency = st.number_input(texts["frequency_label"], min_value=1, max_value=5, value=3, key="direct_freq")
            intensity = st.number_input(texts["intensity_label"], min_value=1, max_value=5, value=3, key="direct_intensity")
            T_value = frequency * intensity
            st.markdown(texts["t_value_text"].format(t_value=T_value, grade=determine_grade(T_value)))
    
    # 개선대책 생성 섹션
    if st.button(texts["generate_improvement_btn"], key="generate_improvement") and activity_text and hazard_text and frequency and intensity and T_value:
        if not api_key_phase2:
            st.warning(texts["api_key_warning"])
        else:
            with st.spinner(texts["generating_improvement"]):
                # 리트리버 풀과 인덱스 확인
                if st.session_state.retriever_pool_df is None or st.session_state.index is None:
                    st.warning(texts["no_data_warning"])
                    # 기본 공통 데이터셋 로드
                    df = load_data("Civil (토목)")
                    retriever_pool_df = df.sample(min(5, len(df)))  # 최대 5개 샘플
                    
                    # 유사 문서는 랜덤 샘플링
                    retrieved_docs = retriever_pool_df.sample(min(3, len(retriever_pool_df)))
                else:
                    # Phase 1에서 구성된 리트리버 풀 및 인덱스 사용
                    retriever_pool_df = st.session_state.retriever_pool_df
                    
                    # 유사 문서 검색 (실제 또는 모의)
                    k_similar = min(3, len(retriever_pool_df))
                    query_text = f"{activity_text} {hazard_text}"
                    query_embedding = embed_texts_with_openai([query_text], api_key=api_key_phase2)[0]
                    query_embedding_array = np.array([query_embedding], dtype='float32')
                    distances, indices = st.session_state.index.search(query_embedding_array, k_similar)
                    retrieved_docs = retriever_pool_df.iloc[indices[0]]
                    
                    # 유사 사례 표시
                    st.markdown(f"#### {texts['similar_cases_header']}")
                    for i, (_, doc) in enumerate(retrieved_docs.iterrows(), 1):
                        # 개선대책 정보 찾기
                        improvement_plan = ""
                        for field in ['개선대책 및 세부관리방안', '개선대책', '개선방안', 'Corrective Action']:
                            if field in doc and pd.notna(doc[field]):
                                improvement_plan = doc[field]
                                break
                        
                        improvement_section = ""
                        if improvement_plan:
                            improvement_title = "개선대책" if st.session_state.language == "Korean" else "Improvement Plan" if st.session_state.language == "English" else "改进措施"
                            improvement_section = f"<strong>{improvement_title}:</strong> {improvement_plan}<br>"
                        
                        st.markdown(f"""
                        <div class="similar-case">
                            <strong>사례 {i}</strong><br>
                            <strong>작업활동:</strong> {doc['작업활동 및 내용']}<br>
                            <strong>유해위험요인:</strong> {doc['유해위험요인 및 환경측면 영향']}<br>
                            <strong>위험도:</strong> 빈도 {doc['빈도']}, 강도 {doc['강도']}, T값 {doc['T']} (등급 {doc['등급']})<br>
                            {improvement_section}
                        </div>
                        """, unsafe_allow_html=True)
                
                # 개선대책 생성 프롬프트 구성
                prompt = construct_prompt_phase2(
                    retrieved_docs, 
                    activity_text, 
                    hazard_text, 
                    frequency, 
                    intensity, 
                    T_value, 
                    target_language
                )
                
                # GPT 호출
                generated_output = generate_with_gpt(prompt, api_key=api_key_phase2, language=target_language)
                
                # 결과 파싱
                parsed_result = parse_gpt_output_phase2(generated_output, language=target_language)
                
                if parsed_result:
                    # 키 이름 매핑
                    key_mappings = {
                        "improvement": "개선대책" if target_language == "Korean" else "improvement_plan" if target_language == "English" else "改进措施",
                        "improved_freq": "개선 후 빈도" if target_language == "Korean" else "improved_frequency" if target_language == "English" else "改进后频率",
                        "improved_intensity": "개선 후 강도" if target_language == "Korean" else "improved_intensity" if target_language == "English" else "改进后强度",
                        "improved_t": "개선 후 T" if target_language == "Korean" else "improved_T" if target_language == "English" else "改进后T值",
                        "reduction_rate": "T 감소율" if target_language == "Korean" else "reduction_rate" if target_language == "English" else "T值降低率"
                    }
                    
                    # 결과 표시
                    improvement_plan = parsed_result.get("improvement", "")
                    improved_freq = parsed_result.get("improved_freq", 1)
                    improved_intensity = parsed_result.get("improved_intensity", 1)
                    improved_T = parsed_result.get("improved_t", improved_freq * improved_intensity)
                    rrr = parsed_result.get("reduction_rate", compute_rrr(T_value, improved_T))
                    
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.markdown(f"#### {texts['improvement_result_header']}")
                    
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        st.markdown(f"##### {texts['improvement_plan_header']}")
                        st.markdown(improvement_plan)
                    
                    with col2:
                        st.markdown(f"##### {texts['risk_improvement_header']}")
                        
                        # 개선 전후 위험도 비교표
                        comparison_df = pd.DataFrame({
                            texts["comparison_columns"][0]: texts["result_table_rows"],
                            texts["comparison_columns"][1]: [frequency, intensity, T_value, determine_grade(T_value)],
                            texts["comparison_columns"][2]: [improved_freq, improved_intensity, improved_T, determine_grade(improved_T)]
                        })
                        st.table(comparison_df)
                        
                        # 위험 감소율 표시
                        st.metric(
                            label=texts["risk_reduction_label"],
                            value=f"{rrr:.2f}%"
                        )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # 위험도 그래프로 표현 (간단한 텍스트 기반 시각화)
                    st.markdown(f"#### {texts['t_value_change_header']}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**{texts['before_improvement']}**")
                        st.progress(T_value / 25)  # 25는 최대 T값
                    
                    with col2:
                        st.markdown(f"**{texts['after_improvement']}**")
                        st.progress(improved_T / 25)
                else:
                    st.error(texts["parsing_error_improvement"])
                    st.write(texts["gpt_response"].format(response=generated_output))

# ----- 푸터 섹션: 로고 이미지 표시 -----
st.markdown('<hr style="margin-top: 50px;">', unsafe_allow_html=True)
st.markdown('<div style="display: flex; justify-content: space-between; align-items: center;">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    if os.path.exists("cau.png"):
        cau_logo = Image.open("cau.png")
        st.image(cau_logo, width=150)
    else:
        st.warning("cau.png 파일을 찾을 수 없습니다.")

with col2:
    if os.path.exists("doosan.png"):
        doosan_logo = Image.open("doosan.png")
        st.image(doosan_logo, width=180)
    else:
        st.warning("doosan.png 파일을 찾을 수 없습니다.")

st.markdown('</div>', unsafe_allow_html=True)
