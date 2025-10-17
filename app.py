import os
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

# SHAP은 첫 호출이 느릴 수 있어요. 단일 샘플만 처리하므로 시간은 오래 걸리지 않습니다.
import shap

from openai import OpenAI

st.set_page_config(page_title="Extubation Prediction Demo", layout="wide")

# =========================
# Config & Secrets
# =========================
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
OPENAI_BASE_URL = st.secrets.get("OPENAI_BASE_URL", None)
DEFAULT_TAGGING_MODEL = st.secrets.get("TAGGING_MODEL", "gpt-4o-mini")
DEFAULT_GENERATION_MODEL = st.secrets.get("GENERATION_MODEL", "gpt-4o-mini")

# =========================
# Utilities
# =========================
REQUIRED_FEATURES = [
    'AGE', 'SEX', 'BMI', 'VENT_DUR', 'CHF', 'CVD', 'CPD', 'CKD', 'CLD',
    'DM', 'ECMO', 'CRRT', 'MAP', 'HR', 'RR', 'BT', 'SPO2', 'GCS', 'PH',
    'PACO2', 'PAO2', 'HCO3', 'LACTATE', 'WBC', 'HB', 'PLT', 'SODIUM',
    'POTASSIUM', 'CHLORIDE', 'BUN', 'CR', 'PT', 'FIO2', 'PEEP', 'PPLAT', 'TV'
]

def _df_from_patient_input(patient_input: dict) -> pd.DataFrame:
    # numpy NaN 허용
    df = pd.DataFrame([patient_input])
    df = df[REQUIRED_FEATURES]  # 순서 고정
    return df

@st.cache_resource(show_spinner=False)
def load_xgb_model(model_path: str = "./models/xgboost_model.json"):
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Upload models/xgboost_model.json to the repo or set external loading."
        )
    booster = xgb.Booster()
    booster.load_model(str(p))
    return booster

def build_openai_client():
    if OPENAI_API_KEY is None:
        return None  # 키가 없어도 룰기반 fallback으로 동작하게 함
    if OPENAI_BASE_URL:
        return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    return OpenAI(api_key=OPENAI_API_KEY)

def rule_based_ontology(df: pd.DataFrame) -> dict:
    """
    LLM을 못 쓸 때를 위한 안전한 fallback. (데모용)
    """
    # 단일 샘플 가정
    row = df.iloc[0]

    hemo = int((row.get("SPO2", 100) < 90) or (row.get("HR", 80) < 40) or (row.get("HR", 80) > 120))
    resp = int(row.get("SPO2", 100) < 85)
    ob   = int(row.get("BMI", 0) >= 30)
    age  = int(row.get("AGE", 0) >= 65)
    airway = int(ob and resp)

    return {
        "patients": [{
            "patient_index": 0,
            "hemodynamic_instability": hemo,
            "respiratory_distress": resp,
            "obesity_risk": ob,
            "age_risk": age,
            "airway_obstruction_risk": airway
        }]
    }

def llm_tag_ontology(client: OpenAI, model_name: str, df: pd.DataFrame) -> dict:
    patient_records = df.to_dict(orient='records')
    prompt = f"""
당신은 의료 전문가입니다. 다음 환자 데이터를 분석하여 발관(extubation) 시 위험 요인이 될 수 있는 온톨로지 특성을 태깅해주세요.

환자 데이터:
{json.dumps(patient_records, ensure_ascii=False, indent=2)}

다음 온톨로지 특성들을 각 환자에 대해 0(해당없음) 또는 1(해당됨)로 판단해주세요:

1. hemodynamic_instability (혈역학적 불안정성)
   - SPO2가 90% 미만이거나 심박수가 비정상적인 경우

2. respiratory_distress (호흡 곤란)
   - SPO2가 85% 미만인 경우

3. obesity_risk (비만 위험)
   - BMI가 30 이상인 경우

4. age_risk (고령 위험)
   - 나이가 65세 이상인 경우

5. airway_obstruction_risk (기도 폐쇄 위험)
   - 비만이면서 동시에 호흡 곤란이 있는 경우

응답은 반드시 다음 JSON 형식으로만 제공해주세요:
{{
  "patients": [
    {{
      "patient_index": 0,
      "hemodynamic_instability": 0,
      "respiratory_distress": 0,
      "obesity_risk": 0,
      "age_risk": 0,
      "airway_obstruction_risk": 0
    }}
  ]
}}
"""
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "의료 데이터 분석 전문가로서, JSON만 반환합니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        response_format={"type": "json_object"}
    )
    return json.loads(resp.choices[0].message.content)

def attach_ontology_features(df: pd.DataFrame, ontology_json: dict):
    feature_names = ["hemodynamic_instability","respiratory_distress","obesity_risk","age_risk","airway_obstruction_risk"]
    for k in feature_names:
        v = ontology_json["patients"][0].get(k, 0)
        df[k] = int(v)
    return df

def run_xgb_predict(booster: xgb.Booster, feature_df: pd.DataFrame):
    dmatrix = xgb.DMatrix(feature_df.values)
    preds = booster.predict(dmatrix)
    p = float(preds[0])
    y = int(p > 0.5)
    return {"probability": p, "class": y}

def compute_shap(booster: xgb.Booster, feature_df: pd.DataFrame):
    dmatrix = xgb.DMatrix(feature_df.values)
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(dmatrix)  # [n_samples, n_features]
    if isinstance(shap_values, list):  # 일부 다중분류 호환
        shap_values = shap_values[0]
    sample_vals = shap_values[0]
    names = feature_df.columns.tolist()
    imp_dict = {names[i]: float(sample_vals[i]) for i in range(len(names))}
    top5 = sorted([(k, abs(v)) for k, v in imp_dict.items()], key=lambda x: x[1], reverse=True)[:5]
    return {"shap_values": shap_values, "feature_importance": imp_dict, "top_risk_factors": top5}

def llm_generate_report(client: OpenAI, model_name: str, patient_input: dict, prediction: dict, shap_exp: dict) -> str:
    top_risk_factors = []
    for k, abs_imp in shap_exp["top_risk_factors"]:
        impact_dir = "위험도 증가" if shap_exp["feature_importance"][k] > 0 else "위험도 감소"
        top_risk_factors.append({"feature_name": k, "importance_score": round(abs_imp, 4), "impact": impact_dir})

    prompt = f"""
당신은 의료진과 환자 보호자 사이의 소통을 돕는 의료 커뮤니케이션 전문가입니다.
아래 인공지능 예측 결과를 바탕으로 보호자가 이해하기 쉬운 설명 레포트를 작성해주세요.

[입력 데이터]
환자 정보:
{json.dumps(patient_input, ensure_ascii=False, indent=2)}

AI 예측 결과:
- 발관 실패 확률: {prediction['probability']:.1%}
- 예측 클래스: {prediction['class']} (0=안전, 1=위험)

주요 위험 요인 (중요도 순위):
{json.dumps(top_risk_factors, ensure_ascii=False, indent=2)}

[레포트 작성 지침]
- 따뜻하고 공감적인 어조
- 전문용어를 쉬운 말로 풀어쓰기
- 확률을 비유로 쉽게 설명
- 상위 3~5개 위험 요인을 구체적으로 풀이
- 마크다운 금지, 섹션 제목은 [섹션명] 형태

반드시 순수 한글 텍스트로만 작성하세요.
"""
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system",
             "content": "당신은 의료 정보를 일반인이 이해하기 쉽게 전달하는 의료 커뮤니케이션 전문가입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6
    )
    body = resp.choices[0].message.content.strip()
    header = f"{'='*60}\n기계환기 발관 안내문\n{'='*60}\n\n생성 일시: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M')}\n\n"
    footer = f"\n\n{'='*60}\n본 안내문은 AI 기반 예측 시스템을 활용하여 작성되었습니다.\n최종 의료 결정은 담당 의료진의 종합적인 판단에 따라 이루어집니다.\n궁금한 점이나 우려사항이 있으시면 언제든 의료진에게 문의해 주세요.\n{'='*60}\n"
    return header + body + footer

# =========================
# UI
# =========================
st.title("Extubation Prediction Demo (Streamlit)")

with st.sidebar:
    st.header("API / 모델 설정")
    use_llm = st.toggle("LLM 사용 (온톨로지 태깅 & 설명문 생성)", value=True)
    tagging_model = st.text_input("태깅 모델명", value=DEFAULT_TAGGING_MODEL)
    generation_model = st.text_input("레포트 생성 모델명", value=DEFAULT_GENERATION_MODEL)
    st.caption("사내 게이트웨이를 쓰면 secrets에 OPENAI_BASE_URL과 모델명을 맞춰주세요.")
    st.markdown("---")
    st.header("모델 파일")
    st.caption("`models/xgboost_model.json`을 레포에 포함하세요. 용량이 크면 Git LFS/외부 스토리지 권장.")
    model_path = st.text_input("XGBoost 모델 경로", "./models/xgboost_model.json")

st.subheader("1) 환자 입력")
colA, colB, colC, colD, colE = st.columns(5)

with colA:
    AGE = st.number_input("AGE", 0, 120, 60)
    SEX = st.selectbox("SEX (0=female,1=male)", [0,1], index=0)
    BMI = st.number_input("BMI", 0.0, 80.0, 31.49)
    VENT_DUR = st.number_input("VENT_DUR (hr)", 0.0, 1000.0, 24.88)
    CHF = st.selectbox("CHF", [0,1], 0)
    CVD = st.selectbox("CVD", [0,1], 1)
    CPD = st.selectbox("CPD", [0,1], 0)
    CKD = st.selectbox("CKD", [0,1], 0)

with colB:
    CLD = st.selectbox("CLD", [0,1], 0)
    DM = st.selectbox("DM", [0,1], 1)
    ECMO = st.selectbox("ECMO", [0,1], 0)
    CRRT = st.selectbox("CRRT", [0,1], 0)
    MAP = st.number_input("MAP", 0.0, 200.0, 65.0)
    HR = st.number_input("HR", 0.0, 220.0, 86.0)
    RR = st.number_input("RR", 0.0, 80.0, 20.0)
    BT = st.number_input("BT (°C, 비어두면  NaN)", value=36.5)

with colC:
    SPO2 = st.number_input("SPO2 (%)", 0.0, 100.0, 95.0)
    GCS = st.number_input("GCS", 3.0, 15.0, 9.0)
    PH = st.number_input("pH", 6.8, 7.8, 7.41, step=0.01)
    PACO2 = st.number_input("PaCO2", 10.0, 120.0, 40.0)
    PAO2 = st.number_input("PaO2", 10.0, 600.0, 136.0)
    HCO3 = st.number_input("HCO3-", 5.0, 45.0, 23.0)
    LACTATE = st.number_input("Lactate", 0.0, 20.0, 1.8)

with colD:
    WBC = st.number_input("WBC", 0.0, 100.0, 13.3)
    HB = st.number_input("Hb", 0.0, 25.0, 11.2)
    PLT = st.number_input("PLT", 0.0, 1000.0, 166.0)
    SODIUM = st.number_input("Na", 100.0, 180.0, 138.0)
    POTASSIUM = st.number_input("K", 1.5, 8.0, 4.1)
    CHLORIDE = st.number_input("Cl", 70.0, 140.0, 105.0)
    BUN = st.number_input("BUN", 0.0, 200.0, 10.0)

with colE:
    CR = st.number_input("Cr", 0.0, 20.0, 0.7)
    PT = st.number_input("PT", 5.0, 30.0, 12.2)
    FIO2 = st.number_input("FiO2 (%)", 21.0, 100.0, 40.0)
    PEEP = st.number_input("PEEP", 0.0, 30.0, 5.0)
    PPLAT = st.number_input("Pplat", 0.0, 60.0, 21.0)
    TV = st.number_input("TV (mL)", 0.0, 1500.0, 546.0)

# NaN 처리 옵션
nan_bt = st.checkbox("BT를 NaN으로 처리", value=False)
if nan_bt:
    BT_val = np.nan
else:
    BT_val = float(BT)

patient_input = {
    'AGE': float(AGE), 'SEX': float(SEX), 'BMI': float(BMI), 'VENT_DUR': float(VENT_DUR),
    'CHF': float(CHF), 'CVD': float(CVD), 'CPD': float(CPD), 'CKD': float(CKD), 'CLD': float(CLD),
    'DM': float(DM), 'ECMO': float(ECMO), 'CRRT': float(CRRT), 'MAP': float(MAP), 'HR': float(HR),
    'RR': float(RR), 'BT': BT_val, 'SPO2': float(SPO2), 'GCS': float(GCS), 'PH': float(PH),
    'PACO2': float(PACO2), 'PAO2': float(PAO2), 'HCO3': float(HCO3), 'LACTATE': float(LACTATE),
    'WBC': float(WBC), 'HB': float(HB), 'PLT': float(PLT), 'SODIUM': float(SODIUM),
    'POTASSIUM': float(POTASSIUM), 'CHLORIDE': float(CHLORIDE), 'BUN': float(BUN), 'CR': float(CR),
    'PT': float(PT), 'FIO2': float(FIO2), 'PEEP': float(PEEP), 'PPLAT': float(PPLAT), 'TV': float(TV)
}

st.subheader("2) 실행")
run = st.button("예측 실행")

if run:
    # 입력 검증
    missing = [f for f in REQUIRED_FEATURES if f not in patient_input]
    if missing:
        st.error(f"필수 피처 누락: {missing}")
        st.stop()

    # 1) 입력 → DF
    df = _df_from_patient_input(patient_input)

    # 2) 온톨로지 태깅 (LLM 또는 룰기반)
    with st.spinner("온톨로지 태깅 중..."):
        try:
            if use_llm and OPENAI_API_KEY:
                client = build_openai_client()
                ontology_json = llm_tag_ontology(client, tagging_model, df)
            else:
                ontology_json = rule_based_ontology(df)
        except Exception as e:
            st.warning(f"LLM 태깅 실패. 룰 기반으로 대체합니다. ({e})")
            ontology_json = rule_based_ontology(df)

    feature_df = attach_ontology_features(df.copy(), ontology_json)

    # 3) 모델 로드 & 예측
    with st.spinner("XGBoost 예측 중..."):
        try:
            booster = load_xgb_model(model_path)
            pred = run_xgb_predict(booster, feature_df)
        except Exception as e:
            st.error(f"예측 오류: {e}")
            st.stop()

    # 4) SHAP
    with st.spinner("SHAP 계산 중..."):
        try:
            shap_exp = compute_shap(booster, feature_df)
        except Exception as e:
            st.warning(f"SHAP 계산 실패: {e}")
            shap_exp = {"feature_importance": {}, "top_risk_factors": []}

    # 5) 레포트 (선택)
    report_text = None
    if use_llm and OPENAI_API_KEY:
        with st.spinner("설명 레포트 생성 중..."):
            try:
                client = build_openai_client()
                report_text = llm_generate_report(client, generation_model, patient_input, pred, shap_exp)
            except Exception as e:
                st.warning(f"레포트 생성 실패: {e}")
                report_text = None

    # =========================
    # 결과 표시
    # =========================
    st.success("완료!")

    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown("### 예측 결과")
        st.metric("발관 실패 확률", f"{pred['probability']*100:.1f}%")
        st.metric("예측 클래스 (0=안전,1=위험)", pred['class'])

        st.markdown("### 온톨로지 태깅 결과")
        st.json(ontology_json)

    with col2:
        st.markdown("### 상위 위험 요인 (SHAP | 절대값 기준 상위 5)")
        if shap_exp["top_risk_factors"]:
            rows = []
            for name, abs_imp in shap_exp["top_risk_factors"]:
                sign = shap_exp["feature_importance"].get(name, 0.0)
                direction = "↑ 위험 증가" if sign > 0 else "↓ 위험 감소"
                rows.append({"feature": name, "abs_importance": round(abs_imp, 5), "direction": direction})
            st.dataframe(pd.DataFrame(rows))
        else:
            st.write("계산되지 않았습니다.")

    if report_text:
        st.markdown("### 보호자 설명용 레포트")
        st.text(report_text)

    st.markdown("### 모델 입력 전체 Feature (추론 시 사용)")
    st.dataframe(feature_df)

    st.caption("※ 본 데모는 교육/연구용입니다. 실제 진료 의사결정에는 반드시 전문의의 판단이 필요합니다.")