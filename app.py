import os
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap
from openai import OpenAI

# -------------------------------
# 페이지/키/모델 경로
# -------------------------------
st.set_page_config(page_title="🖥️ KCD 2025 J. - Will the first extubation be successful?", layout="wide")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
OPENAI_BASE_URL = st.secrets.get("OPENAI_BASE_URL", None)

USE_LLM = True
FIXED_TAGGING_MODEL = "gpt-4o-mini"
FIXED_GENERATION_MODEL = "gpt-4o-mini"
CHAT_MODEL = "gpt-4o-mini"

# ✅ 온톨로지 포함 학습 모델
FIXED_MODEL_PATH = "./models/best_model_medgemma.pkl"

# -------------------------------
# 피처 스키마
# -------------------------------
REQUIRED_FEATURES = [
    'AGE', 'SEX', 'BMI', 'VENT_DUR', 'CHF', 'CVD', 'CPD', 'CKD', 'CLD',
    'DM', 'ECMO', 'CRRT', 'MAP', 'HR', 'RR', 'BT', 'SPO2', 'GCS', 'PH',
    'PACO2', 'PAO2', 'HCO3', 'LACTATE', 'WBC', 'HB', 'PLT', 'SODIUM',
    'POTASSIUM', 'CHLORIDE', 'BUN', 'CR', 'PT', 'FIO2', 'PEEP', 'PPLAT', 'TV'
]

# ✅ 온톨로지 10개 (모델 입력에도 포함됨)
ONTO_FEATURES = [
    "diabetes_mellitus",
    "obesity",
    "prolonged_mechanical_ventilation_history",
    "advanced_age",
    "low_PaO2_FiO2_ratio",
    "congestive_heart_failure",
    "anemia",
    "hemodynamic_instability",
    "low_mean_arterial_pressure",
    "leukocytosis",
]

ALL_FEATURES_FALLBACK = REQUIRED_FEATURES + ONTO_FEATURES

# -------------------------------
# 예시 케이스
# -------------------------------
EXAMPLE_CASES = {
    "케이스 1": {'AGE': 60.0,'SEX': 0.0,'BMI': 31.49090228,'VENT_DUR': 24.88333333,'CHF': 0.0,'CVD': 1.0,'CPD': 0.0,'CKD': 0.0,'CLD': 0.0,'DM': 1.0,'ECMO': 0.0,'CRRT': 0.0,'MAP': 65.0,'HR': 86.0,'RR': 20.0,'BT': 36.5,'SPO2': 95.0,'GCS': 9.0,'PH': 7.41,'PACO2': 40.0,'PAO2': 136.0,'HCO3': 23.0,'LACTATE': 1.8,'WBC': 13.3,'HB': 11.2,'PLT': 166.0,'SODIUM': 138.0,'POTASSIUM': 4.1,'CHLORIDE': 105.0,'BUN': 10.0,'CR': 0.7,'PT': 12.2,'FIO2': 40.0,'PEEP': 5.0,'PPLAT': 21.0,'TV': 546.0},
    "케이스 2": {'AGE': 72.0,'SEX': 1.0,'BMI': 27.3,'VENT_DUR': 72.0,'CHF': 1.0,'CVD': 0.0,'CPD': 1.0,'CKD': 0.0,'CLD': 0.0,'DM': 0.0,'ECMO': 0.0,'CRRT': 0.0,'MAP': 58.0,'HR': 128.0,'RR': 28.0,'BT': 38.4,'SPO2': 88.0,'GCS': 8.0,'PH': 7.32,'PACO2': 50.0,'PAO2': 70.0,'HCO3': 24.0,'LACTATE': 3.2,'WBC': 18.5,'HB': 10.1,'PLT': 140.0,'SODIUM': 136.0,'POTASSIUM': 4.8,'CHLORIDE': 104.0,'BUN': 26.0,'CR': 1.2,'PT': 14.0,'FIO2': 60.0,'PEEP': 8.0,'PPLAT': 28.0,'TV': 420.0},
    "케이스 3": {'AGE': 45.0,'SEX': 1.0,'BMI': 33.8,'VENT_DUR': 12.0,'CHF': 0.0,'CVD': 0.0,'CPD': 0.0,'CKD': 0.0,'CLD': 0.0,'DM': 0.0,'ECMO': 0.0,'CRRT': 0.0,'MAP': 75.0,'HR': 92.0,'RR': 18.0,'BT': 36.8,'SPO2': 97.0,'GCS': 13.0,'PH': 7.43,'PACO2': 37.0,'PAO2': 120.0,'HCO3': 24.0,'LACTATE': 1.2,'WBC': 9.8,'HB': 12.5,'PLT': 210.0,'SODIUM': 140.0,'POTASSIUM': 3.9,'CHLORIDE': 103.0,'BUN': 12.0,'CR': 0.8,'PT': 12.0,'FIO2': 35.0,'PEEP': 5.0,'PPLAT': 19.0,'TV': 500.0}
}

# -------------------------------
# 클라이언트/모델 로더
# -------------------------------
def build_openai_client():
    if OPENAI_API_KEY is None:
        return None
    if OPENAI_BASE_URL:
        return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    return OpenAI(api_key=OPENAI_API_KEY)

@st.cache_resource(show_spinner=False)
def load_model(model_path: str = FIXED_MODEL_PATH):
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model not found at {model_path}.")
    model = joblib.load(str(p))  # RandomForest or Pipeline with ColumnTransformer
    return model

def _df_from_patient_input(patient_input: dict) -> pd.DataFrame:
    df = pd.DataFrame([patient_input])
    df = df[REQUIRED_FEATURES]
    return df

# -------------------------------
# 온톨로지 라벨/설명
# -------------------------------
def _ontology_label_maps():
    labels = {
        "diabetes_mellitus": "당뇨병",
        "obesity": "비만",
        "prolonged_mechanical_ventilation_history": "장기간 기계환기 병력",
        "advanced_age": "고령",
        "low_PaO2_FiO2_ratio": "낮은 PaO2/FiO2 비율",
        "congestive_heart_failure": "울혈성 심부전",
        "anemia": "빈혈",
        "hemodynamic_instability": "혈역학적 불안정",
        "low_mean_arterial_pressure": "낮은 평균 동맥압",
        "leukocytosis": "백혈구 증가증",
    }
    desc = {
        "diabetes_mellitus": "당뇨병 병력 또는 혈당 조절 문제",
        "obesity": "BMI≥30",
        "prolonged_mechanical_ventilation_history": "기계환기 48시간 이상",
        "advanced_age": "나이≥65세",
        "low_PaO2_FiO2_ratio": "PaO2/FiO2<200",
        "congestive_heart_failure": "울혈성 심부전 병력",
        "anemia": "Hb<10 g/dL",
        "hemodynamic_instability": "HR<40/HR>120 또는 MAP<60",
        "low_mean_arterial_pressure": "MAP<65",
        "leukocytosis": "WBC>12 (×10^3/μL)",
    }
    return labels, desc

def summarize_ontology_for_report(ontology_json: dict):
    labels, desc = _ontology_label_maps()
    row = ontology_json["patients"][0]
    positives, negatives = [], []
    for k in labels:
        item = {
            "key": k,
            "name": labels[k],
            "rule": desc[k],
            "value": int(row.get(k, 0))
        }
        (positives if item["value"] == 1 else negatives).append(item)
    return positives, negatives

def ontology_pretty_table(ontology_json: dict) -> pd.DataFrame:
    labels, desc = _ontology_label_maps()
    row = ontology_json["patients"][0]
    rows = []
    for k in labels:
        val = int(row.get(k, 0))
        icon = "✅" if val == 1 else "❌"
        rows.append({"특성": labels[k], "설명": desc[k], "여부": icon})
    return pd.DataFrame(rows)

# -------------------------------
# 온톨로지 태깅 (LLM/룰)
# -------------------------------
def rule_based_ontology(df: pd.DataFrame) -> dict:
    row = df.iloc[0]
    dm = int(row.get("DM", 0) == 1)
    obesity = int(float(row.get("BMI", 0)) >= 30)
    prolonged_mv = int(float(row.get("VENT_DUR", 0)) >= 48)
    advanced_age = int(float(row.get("AGE", 0)) >= 65)

    pao2 = float(row.get("PAO2", 0))
    fio2_pct = float(row.get("FIO2", 21.0))
    fio2_frac = max(fio2_pct, 21.0) / 100.0
    pf_ratio = pao2 / fio2_frac if fio2_frac > 0 else 0.0
    low_pfr = int(pf_ratio < 200)

    chf = int(row.get("CHF", 0) == 1)
    anemia = int(float(row.get("HB", 0)) < 10.0)
    hemo_instab = int((float(row.get("HR", 80)) < 40) or (float(row.get("HR", 80)) > 120) or (float(row.get("MAP", 80)) < 60))
    low_map = int(float(row.get("MAP", 80)) < 65)
    leukocytosis = int(float(row.get("WBC", 0)) > 12.0)

    return {
        "patients": [
            {
                "patient_index": 0,
                "diabetes_mellitus": dm,
                "obesity": obesity,
                "prolonged_mechanical_ventilation_history": prolonged_mv,
                "advanced_age": advanced_age,
                "low_PaO2_FiO2_ratio": low_pfr,
                "congestive_heart_failure": chf,
                "anemia": anemia,
                "hemodynamic_instability": hemo_instab,
                "low_mean_arterial_pressure": low_map,
                "leukocytosis": leukocytosis,
            }
        ]
    }

def llm_tag_ontology(client: OpenAI, df: pd.DataFrame) -> dict:
    patient_records = df.to_dict(orient='records')
    schema = {
        "patients": [{
            "patient_index": 0,
            "diabetes_mellitus": 0,
            "obesity": 0,
            "prolonged_mechanical_ventilation_history": 0,
            "advanced_age": 0,
            "low_PaO2_FiO2_ratio": 0,
            "congestive_heart_failure": 0,
            "anemia": 0,
            "hemodynamic_instability": 0,
            "low_mean_arterial_pressure": 0,
            "leukocytosis": 0
        }]
    }
    prompt = (
        "당신은 의료 전문가입니다. 다음 환자 데이터를 분석하여 발관(extubation) 시 위험 요인이 될 수 있는 "
        "온톨로지 특성을 태깅해주세요.\n\n"
        f"환자 데이터:\n{json.dumps(patient_records, ensure_ascii=False, indent=2)}\n\n"
        "다음 특성을 0 또는 1로 반환:\n"
        "1. diabetes_mellitus\n2. obesity\n3. prolonged_mechanical_ventilation_history\n"
        "4. advanced_age\n5. low_PaO2_FiO2_ratio\n6. congestive_heart_failure\n7. anemia\n"
        "8. hemodynamic_instability\n9. low_mean_arterial_pressure\n10. leukocytosis\n\n"
        "반드시 아래 JSON 스키마만 반환:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )
    resp = client.chat.completions.create(
        model=FIXED_TAGGING_MODEL,
        messages=[
            {"role":"system","content":"의료 데이터 분석 전문가로서 JSON만 반환합니다."},
            {"role":"user","content":prompt}
        ],
        temperature=0.2,
        response_format={"type":"json_object"}
    )
    return json.loads(resp.choices[0].message.content)

def attach_ontology_features(df: pd.DataFrame, ontology_json: dict):
    row = ontology_json["patients"][0]
    for k in ONTO_FEATURES:
        df[k] = int(row.get(k, 0))
    return df

# -------------------------------
# 모델 기대 피처 자동 추론
# -------------------------------
def get_expected_model_features(model, fallback_cols):
    # 1) feature_names_in_
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    # 2) 파이프라인 내부 추론
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        if isinstance(model, Pipeline):
            for name, step in model.steps:
                if hasattr(step, "feature_names_in_"):
                    return list(step.feature_names_in_)
                if isinstance(step, ColumnTransformer) and hasattr(step, "feature_names_in_"):
                    return list(step.feature_names_in_)
    except Exception:
        pass
    # 3) 실패 시 폴백(원본+온톨로지)
    return list(fallback_cols)

# -------------------------------
# 예측 / SHAP
# -------------------------------
def run_predict(model, df_model: pd.DataFrame):
    # DataFrame 우선
    try:
        proba = model.predict_proba(df_model)
    except Exception:
        proba = model.predict_proba(df_model.values)

    pos_idx = 1
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
        if 1 in classes:
            pos_idx = classes.index(1)

    p = float(proba[0, pos_idx])
    y = "위험" if p > 0.5 else "안전"
    return {"probability": p, "class_label": y}

def compute_shap(model, df_model: pd.DataFrame):
    """
    안정형 SHAP: KernelExplainer로 class1 확률에 대한 기여도를 계산.
    파이프라인/희소행렬/트리해석기 불일치 문제를 회피.
    """
    # class1 인덱스
    pos_idx = 1
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
        if 1 in classes:
            pos_idx = classes.index(1)

    # 예측 함수: class1 확률 반환
    def pred_fn(X):
        X_df = pd.DataFrame(X, columns=df_model.columns)
        try:
            proba = model.predict_proba(X_df)
        except Exception:
            proba = model.predict_proba(X_df.values)
        return proba[:, pos_idx]

    # 백그라운드: 동일 샘플 복제(속도/안정 균형)
    bg = df_model.to_numpy()
    if len(bg) < 10:
        bg = np.vstack([bg] * 10)

    explainer = shap.KernelExplainer(pred_fn, bg)
    shap_arr = explainer.shap_values(df_model.to_numpy(), nsamples=100)
    shap_arr = np.asarray(shap_arr)
    sample_vals = shap_arr[0]  # 1D (n_features,)

    names = df_model.columns.tolist()
    imp_dict = {names[i]: float(np.asarray(sample_vals[i]).reshape(())) for i in range(len(names))}
    top5 = sorted([(k, abs(v)) for k, v in imp_dict.items()], key=lambda x: x[1], reverse=True)[:5]

    return {"shap_values": shap_arr, "feature_importance": imp_dict, "top_risk_factors": top5}

# -------------------------------
# 레포트 생성
# -------------------------------
def summarize_shap_for_report(shap_exp: dict, top_k: int = 5):
    fi = shap_exp.get("feature_importance", {}) or {}
    if not fi:
        return [], []
    triples = [(name, abs(val), val) for name, val in fi.items()]
    triples.sort(key=lambda x: x[1], reverse=True)
    top = [{"feature": n, "abs_importance": round(a, 5), "direction": ("위험 증가" if s > 0 else "위험 감소")}
           for (n, a, s) in triples[:top_k]]
    bottom = [{"feature": n, "abs_importance": round(a, 5), "direction": ("위험 증가" if s > 0 else "위험 감소")}
              for (n, a, s) in triples[-top_k:]]
    return top, bottom

def llm_generate_report(client: OpenAI, patient_input: dict, prediction: dict, shap_exp: dict, ontology_json: dict) -> str:
    ont_pos, ont_neg = summarize_ontology_for_report(ontology_json)
    shap_top, _ = summarize_shap_for_report(shap_exp, top_k=5)

    ont_pos_for_llm = [{"name": x["name"], "rule": x["rule"]} for x in ont_pos] or [{"name": "해당 항목 없음", "rule": ""}]
    ont_neg_for_llm = [{"name": x["name"], "rule": x["rule"]} for x in ont_neg]
    shap_top_for_llm = shap_top or [{"feature": "해당 없음", "abs_importance": 0.0, "direction": ""}]
    cls_label = prediction.get("class_label", "안전")

    prompt = (
        "당신은 의료진과 환자 보호자 사이의 소통을 돕는 의료 커뮤니케이션 전문가입니다.\n"
        "아래 '모델 예측', '온톨로지 판단 결과', 'SHAP 중요 요인'을 모두 참고하여, "
        "보호자가 이해하기 쉬운 설명 레포트를 영어로 작성하세요.\n\n"
        f"[모델 예측]\n- 발관 실패 확률: {prediction['probability']:.1%}\n- 예측 클래스(안전/위험): {cls_label}\n\n"
        f"[온톨로지 판단 결과]\n- 이 환자에게 실제 해당된 항목(값=1): {json.dumps(ont_pos_for_llm, ensure_ascii=False)}\n"
        f"- 해당되지 않은 항목(값=0): {json.dumps(ont_neg_for_llm, ensure_ascii=False)}\n\n"
        f"[SHAP 중요 요인 Top 5]\n- {json.dumps(shap_top_for_llm, ensure_ascii=False)}\n\n"
        f"[입력 데이터(요약)]\n- 일부 주요 수치: "
        f"{json.dumps({k: patient_input[k] for k in ['AGE','BMI','SPO2','MAP','HR','RR','GCS','PH','PACO2','PAO2','HCO3','LACTATE','FIO2','PEEP','PPLAT','TV'] if k in patient_input}, ensure_ascii=False)}\n\n"
        "[작성 지침]\n"
        "- 톤&스타일: 따뜻하고 공감적인 어조를 유지하세요. 보호자의 불안감을 이해하면서도 정확한 정보를 전달하세요. 의학적 전문성과 인간미를 균형 있게 표현하세요."
        "- 온톨로지에서 값=1인 항목은 '이 환자에게 실제로 관찰된 위험 신호'로 반드시 본문에 포함하세요.\n"
        "- SHAP 상위 요인은 '왜 이런 예측이 나왔는지' 설명하는 근거로 사용하세요. (증가/감소 방향을 자연스럽게 서술)\n"
        "- 값=0인 온톨로지 항목은 필요 시 '완화 요인' 또는 '현재는 해당 없음'으로 간단히 언급해도 됩니다.\n"
        "- 확률 수치는 비유/사례로 쉽게 설명하되, 절대적인 진단이 아님을 분명히 하세요.\n"
        "- 전문용어는 쉬운 말로 풀어서 설명하세요.\n"
        "- 마크다운 금지, 섹션 제목은 [섹션명] 형태.\n"
        "- 분량: 8~14문장 정도.\n\n"
        "[권장 구성]\n"
        "1) [인사 및 목적] 2~3문장\n"
        "2) [현재 상태 요약] 2~3문장 (중요 수치 간단 해석)\n"
        "3) [예측 결과 해석] 2~3문장 (확률 의미, 안전/위험 맥락)\n"
        "4) [해당된 위험 신호(온톨로지)] 2~4문장 (값=1 항목을 풀어서 설명)\n"
        "5) [예측 근거(모델 관점)] 2~3문장 (SHAP Top 요인을 쉬운 언어로)\n"
        "6) [권고 및 마무리] 1~2문장 (모니터링/소통 강조)\n"
    )

    resp = client.chat.completions.create(
        model=FIXED_GENERATION_MODEL,
        messages=[
            {"role":"system","content":"당신은 의료 정보를 일반인이 이해하기 쉽게 전달하는 의료 커뮤니케이션 전문가입니다."},
            {"role":"user","content":prompt}
        ],
        temperature=0.5
    )

    body = resp.choices[0].message.content.strip()
    header = f"{'='*60}\n기계환기 발관 안내문\n{'='*60}\n\n생성 일시: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M')}\n\n"
    footer = f"\n\n{'='*60}\n본 안내문은 AI 기반 예측 시스템을 활용하여 작성되었습니다.\n최종 의료 결정은 담당 의료진의 종합적인 판단에 따라 이루어집니다.\n궁금한 점이나 우려사항이 있으시면 언제든 의료진에게 문의해 주세요.\n{'='*60}\n"
    return header + body + footer

# -------------------------------
# 세션 상태
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state:
    st.session_state.memory = {}

# -------------------------------
# UI
# -------------------------------
st.title("🖥️ KCD 2025 J. - Will the first extubation be successful?")

st.subheader("➡️ Select Example Case")
selected_case = st.selectbox("Example Case", list(EXAMPLE_CASES.keys()), index=0)
case_vals = EXAMPLE_CASES[selected_case]

# 초기값 주입
for k in REQUIRED_FEATURES:
    if f"val_{k}" not in st.session_state:
        st.session_state[f"val_{k}"] = case_vals.get(k, np.nan)

def apply_case(vals: dict):
    for k, v in vals.items():
        st.session_state[f"val_{k}"] = v

if st.button("Loading the values from the selected case..."):
    apply_case(EXAMPLE_CASES[selected_case])
    st.success(f"{selected_case} values have been loaded into the input form.")

st.subheader("🗒️ Patient Variables input")
colA, colB, colC, colD, colE = st.columns(5)

with colA:
    st.session_state["val_AGE"] = st.number_input("AGE", 0, 120, int(st.session_state["val_AGE"]))
    st.session_state["val_SEX"] = st.selectbox("SEX (0=female,1=male)", [0,1], index=int(st.session_state["val_SEX"]))
    st.session_state["val_BMI"] = st.number_input("BMI", 0.0, 80.0, float(st.session_state["val_BMI"]))
    st.session_state["val_VENT_DUR"] = st.number_input("VENT_DUR (hr)", 0.0, 1000.0, float(st.session_state["val_VENT_DUR"]))
    st.session_state["val_CHF"] = st.selectbox("CHF", [0,1], index=int(st.session_state["val_CHF"]))
    st.session_state["val_CVD"] = st.selectbox("CVD", [0,1], index=int(st.session_state["val_CVD"]))
    st.session_state["val_CPD"] = st.selectbox("CPD", [0,1], index=int(st.session_state["val_CPD"]))
    st.session_state["val_CKD"] = st.selectbox("CKD", [0,1], index=int(st.session_state["val_CKD"]))

with colB:
    st.session_state["val_CLD"] = st.selectbox("CLD", [0,1], index=int(st.session_state["val_CLD"]))
    st.session_state["val_DM"] = st.selectbox("DM", [0,1], index=int(st.session_state["val_DM"]))
    st.session_state["val_ECMO"] = st.selectbox("ECMO", [0,1], index=int(st.session_state["val_ECMO"]))
    st.session_state["val_CRRT"] = st.selectbox("CRRT", [0,1], index=int(st.session_state["val_CRRT"]))
    st.session_state["val_MAP"] = st.number_input("MAP", 0.0, 200.0, float(st.session_state["val_MAP"]))
    st.session_state["val_HR"] = st.number_input("HR", 0.0, 220.0, float(st.session_state["val_HR"]))
    st.session_state["val_RR"] = st.number_input("RR", 0.0, 80.0, float(st.session_state["val_RR"]))
    st.session_state["val_BT"] = st.number_input("BT (°C)", value=float(case_vals.get("BT", 36.5)))

with colC:
    st.session_state["val_SPO2"] = st.number_input("SPO2 (%)", 0.0, 100.0, float(st.session_state["val_SPO2"]))
    st.session_state["val_GCS"] = st.number_input("GCS", 3.0, 15.0, float(st.session_state["val_GCS"]))
    st.session_state["val_PH"] = st.number_input("pH", 6.8, 7.8, float(st.session_state["val_PH"]), step=0.01)
    st.session_state["val_PACO2"] = st.number_input("PaCO2", 10.0, 120.0, float(st.session_state["val_PACO2"]))
    st.session_state["val_PAO2"] = st.number_input("PaO2", 10.0, 600.0, float(st.session_state["val_PAO2"]))
    st.session_state["val_HCO3"] = st.number_input("HCO3-", 5.0, 45.0, float(st.session_state["val_HCO3"]))
    st.session_state["val_LACTATE"] = st.number_input("Lactate", 0.0, 20.0, float(st.session_state["val_LACTATE"]))

with colD:
    st.session_state["val_WBC"] = st.number_input("WBC", 0.0, 100.0, float(st.session_state["val_WBC"]))
    st.session_state["val_HB"] = st.number_input("Hb", 0.0, 25.0, float(st.session_state["val_HB"]))
    st.session_state["val_PLT"] = st.number_input("PLT", 0.0, 1000.0, float(st.session_state["val_PLT"]))
    st.session_state["val_SODIUM"] = st.number_input("Na", 100.0, 180.0, float(st.session_state["val_SODIUM"]))
    st.session_state["val_POTASSIUM"] = st.number_input("K", 1.5, 8.0, float(st.session_state["val_POTASSIUM"]))
    st.session_state["val_CHLORIDE"] = st.number_input("Cl", 70.0, 140.0, float(st.session_state["val_CHLORIDE"]))
    st.session_state["val_BUN"] = st.number_input("BUN", 0.0, 200.0, float(st.session_state["val_BUN"]))

with colE:
    st.session_state["val_CR"] = st.number_input("Cr", 0.0, 20.0, float(st.session_state["val_CR"]))
    st.session_state["val_PT"] = st.number_input("PT", 5.0, 30.0, float(st.session_state["val_PT"]))
    st.session_state["val_FIO2"] = st.number_input("FiO2 (%)", 21.0, 100.0, float(st.session_state["val_FIO2"]))
    st.session_state["val_PEEP"] = st.number_input("PEEP", 0.0, 30.0, float(st.session_state["val_PEEP"]))
    st.session_state["val_PPLAT"] = st.number_input("Pplat", 0.0, 60.0, float(st.session_state["val_PPLAT"]))
    st.session_state["val_TV"] = st.number_input("TV (mL)", 0.0, 1500.0, float(st.session_state["val_TV"]))

st.subheader("➡️ Generate Prediction & Report")
run = st.button("Check Prediction Results & Report")

if run:
    # 1) 입력 DF(베이스라인)
    patient_input = {k: float(st.session_state[f"val_{k}"]) if not np.isnan(st.session_state[f"val_{k}"]) else np.nan
                     for k in REQUIRED_FEATURES}
    df_base = _df_from_patient_input(patient_input)

    # 2) 온톨로지 태깅 → 모델/UI 둘 다 사용
    with st.spinner("🤖 LLM agent is tagging Ontologies..."):
        try:
            if USE_LLM and OPENAI_API_KEY:
                client = build_openai_client()
                ontology_json = llm_tag_ontology(client, df_base)
            else:
                ontology_json = rule_based_ontology(df_base)
        except Exception as e:
            st.warning(f"LLM 태깅 실패. 룰 기반으로 대체합니다. ({e})")
            ontology_json = rule_based_ontology(df_base)

    # ✅ 모델 입력/표시 모두 온톨로지 포함
    df_with_onto = attach_ontology_features(df_base.copy(), ontology_json)

    # 3) 모델 로드 & 예측 (온톨로지 포함 입력)
    with st.spinner("🤖 Predicting with Random Forest..."):
        try:
            model = load_model(FIXED_MODEL_PATH)
            expected_cols = get_expected_model_features(model, fallback_cols=ALL_FEATURES_FALLBACK)
            df_model = df_with_onto.reindex(columns=expected_cols)
            pred = run_predict(model, df_model)
        except Exception as e:
            st.error(f"예측 오류: {e}")
            st.stop()

    # 4) SHAP (온톨로지 포함 입력 기준)
    with st.spinner("SHAP calculation in progress..."):
        try:
            shap_exp = compute_shap(model, df_model)
        except Exception as e:
            st.warning(f"SHAP 계산 실패: {e}")
            shap_exp = {"feature_importance": {}, "top_risk_factors": []}

    # 5) 레포트
    report_text = None
    if USE_LLM and OPENAI_API_KEY:
        with st.spinner("🫶 Generating explanation report for guardians..."):
            try:
                client = build_openai_client()
                report_text = llm_generate_report(client, patient_input, pred, shap_exp, ontology_json)
            except Exception as e:
                st.warning(f"Report generation failed: {e}")
                report_text = None

    # 결과 표시
    st.success("Complete!")

    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown("### Prediction Results")
        st.metric("Extubation Failure Probability", f"{pred['probability']*100:.1f}%")
        st.metric("Predicted Class", pred["class_label"])

        st.markdown("### Ontology Tagging Results")
        pretty_df = ontology_pretty_table(ontology_json)
        st.dataframe(pretty_df, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("### Top Risk Factors (SHAP | Absolute Top 5)")
        if "top_risk_factors" in shap_exp and shap_exp["top_risk_factors"]:
            rows = []
            for name, abs_imp in shap_exp["top_risk_factors"]:
                sign = shap_exp["feature_importance"].get(name, 0.0)
                direction = "↑ 위험 증가" if sign > 0 else "↓ 위험 감소"
                rows.append({"feature": name, "abs_importance": round(abs_imp, 5), "direction": direction})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.write("계산되지 않았습니다.")

    # 보호자 설명용 레포트 (편집/다운로드)
    st.markdown("### Explanation Report for Guardians")
    if report_text:
        if "report_text" not in st.session_state or not st.session_state.get("report_text"):
            st.session_state.report_text = report_text
        st.session_state.report_text = st.text_area(
            "Generated Report (Editable)",
            value=st.session_state.report_text,
            height=420,
            help="필요 시 문구를 수정하여 보호자 커뮤니케이션에 활용하세요."
        )
        st.download_button(
            label="Download Report .txt",
            data=st.session_state.report_text,
            file_name=f"extubation_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    else:
        st.info("Report has not been generated. Please run the prediction above to generate it.")

    # ▶ 모델 입력 전체 Feature (온톨로지 포함)
    st.markdown("### Every Features")
    # 보기 좋게: 모델 입력 피처 + 온톨로지 라벨/설명 합쳐서 보여주기
    labels, desc = _ontology_label_maps()
    onto_row = ontology_json["patients"][0]
    df_display = df_model.T.rename(columns={0: "Value"})
    sep = pd.DataFrame([[""]], index=[" "], columns=["Value"])
    onto_rows = []
    for key in ONTO_FEATURES:
        onto_rows.append({
            "Feature": key,
            "태깅결과(0/1)": int(onto_row.get(key, 0)),
            "온톨로지_특성": labels.get(key, key)
        })
    onto_df = pd.DataFrame(onto_rows).set_index("Feature")
    df_display = pd.concat([df_display, sep, onto_df], axis=0)
    st.dataframe(df_display, use_container_width=True, height=600)

    # 채팅 컨텍스트 메모리
    st.session_state.memory = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "selected_case": selected_case,
        "patient_input": patient_input,
        "prediction": pred,
        "shap_top": shap_exp.get("top_risk_factors", []),
    }

# -------------------------------
# 사이드바: 챗봇
# -------------------------------
with st.sidebar:
    from pathlib import Path
    image_path = Path("image_kcd.jpg")
    if image_path.exists():
        st.image(str(image_path), use_container_width=True)

    st.header("💬 Chatbot Assistant for Patient Guardians")
    if OPENAI_API_KEY is None:
        st.caption("OpenAI 키가 없어서 채팅은 비활성화됩니다. (Secrets에 OPENAI_API_KEY 추가)")
    else:
        context_blob = json.dumps(st.session_state.memory, ensure_ascii=False, indent=2) if st.session_state.memory else "최근 예측 컨텍스트 없음."
        system_msg = (
            "당신은 중환자실에 입실한 환자 보호자를 대하는 의료인입니다. "
            "아래 '최근 예측 컨텍스트'를 참고하여 친절하고 쉽게 답변하세요. 의료인이 아닌 사람들도 알아들을 수 있도록 설명하세요. 영어로 응답하세요. \n\n"
            f"[최근 예측 컨텍스트]\n{context_blob}"
        )
        for m in st.session_state.chat_history:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        user_msg = st.chat_input("무엇이든 물어보세요")
        if user_msg:
            st.session_state.chat_history.append({"role":"user","content":user_msg})
            with st.chat_message("user"):
                st.markdown(user_msg)
            try:
                client = build_openai_client()
                messages = [{"role":"system","content":system_msg}] + st.session_state.chat_history[-20:]
                resp = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=messages,
                    temperature=0.3
                )
                bot_text = resp.choices[0].message.content.strip()
            except Exception as e:
                bot_text = f"(오류) {e}"
            st.session_state.chat_history.append({"role":"assistant","content":bot_text})
            with st.chat_message("assistant"):
                st.markdown(bot_text)