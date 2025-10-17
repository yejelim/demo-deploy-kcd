import os
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import shap
from openai import OpenAI

# -------------------------------
# 고정 설정(사이드바 입력 제거 버전)
# -------------------------------
st.set_page_config(page_title="Extubation Prediction Demo", layout="wide")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
OPENAI_BASE_URL = st.secrets.get("OPENAI_BASE_URL", None)

# LLM 사용 여부와 모델명/경로를 코드에서 고정
USE_LLM = True   # secrets에 키가 없으면 자동 Fallback
FIXED_TAGGING_MODEL = "gpt-4o-mini"        # 사내 게이트웨이 쓰면 거기서 지원하는 이름으로 교체
FIXED_GENERATION_MODEL = "gpt-4o-mini"
CHAT_MODEL = "gpt-4o-mini"                 # 사이드바 채팅용
FIXED_MODEL_PATH = "./models/xgboost_model.json"

# -------------------------------
# 유틸 / 상수
# -------------------------------
REQUIRED_FEATURES = [
    'AGE', 'SEX', 'BMI', 'VENT_DUR', 'CHF', 'CVD', 'CPD', 'CKD', 'CLD',
    'DM', 'ECMO', 'CRRT', 'MAP', 'HR', 'RR', 'BT', 'SPO2', 'GCS', 'PH',
    'PACO2', 'PAO2', 'HCO3', 'LACTATE', 'WBC', 'HB', 'PLT', 'SODIUM',
    'POTASSIUM', 'CHLORIDE', 'BUN', 'CR', 'PT', 'FIO2', 'PEEP', 'PPLAT', 'TV'
]

# 예시 케이스 1/2/3
EXAMPLE_CASES = {
    "케이스 1": {'AGE': 60.0,'SEX': 0.0,'BMI': 31.49090228,'VENT_DUR': 24.88333333,'CHF': 0.0,'CVD': 1.0,'CPD': 0.0,'CKD': 0.0,'CLD': 0.0,'DM': 1.0,'ECMO': 0.0,'CRRT': 0.0,'MAP': 65.0,'HR': 86.0,'RR': 20.0,'BT': 36.5,'SPO2': 95.0,'GCS': 9.0,'PH': 7.41,'PACO2': 40.0,'PAO2': 136.0,'HCO3': 23.0,'LACTATE': 1.8,'WBC': 13.3,'HB': 11.2,'PLT': 166.0,'SODIUM': 138.0,'POTASSIUM': 4.1,'CHLORIDE': 105.0,'BUN': 10.0,'CR': 0.7,'PT': 12.2,'FIO2': 40.0,'PEEP': 5.0,'PPLAT': 21.0,'TV': 546.0},
    "케이스 2": {'AGE': 72.0,'SEX': 1.0,'BMI': 27.3,'VENT_DUR': 72.0,'CHF': 1.0,'CVD': 0.0,'CPD': 1.0,'CKD': 0.0,'CLD': 0.0,'DM': 0.0,'ECMO': 0.0,'CRRT': 0.0,'MAP': 58.0,'HR': 128.0,'RR': 28.0,'BT': 38.4,'SPO2': 88.0,'GCS': 8.0,'PH': 7.32,'PACO2': 50.0,'PAO2': 70.0,'HCO3': 24.0,'LACTATE': 3.2,'WBC': 18.5,'HB': 10.1,'PLT': 140.0,'SODIUM': 136.0,'POTASSIUM': 4.8,'CHLORIDE': 104.0,'BUN': 26.0,'CR': 1.2,'PT': 14.0,'FIO2': 60.0,'PEEP': 8.0,'PPLAT': 28.0,'TV': 420.0},
    "케이스 3": {'AGE': 45.0,'SEX': 1.0,'BMI': 33.8,'VENT_DUR': 12.0,'CHF': 0.0,'CVD': 0.0,'CPD': 0.0,'CKD': 0.0,'CLD': 0.0,'DM': 0.0,'ECMO': 0.0,'CRRT': 0.0,'MAP': 75.0,'HR': 92.0,'RR': 18.0,'BT': 36.8,'SPO2': 97.0,'GCS': 13.0,'PH': 7.43,'PACO2': 37.0,'PAO2': 120.0,'HCO3': 24.0,'LACTATE': 1.2,'WBC': 9.8,'HB': 12.5,'PLT': 210.0,'SODIUM': 140.0,'POTASSIUM': 3.9,'CHLORIDE': 103.0,'BUN': 12.0,'CR': 0.8,'PT': 12.0,'FIO2': 35.0,'PEEP': 5.0,'PPLAT': 19.0,'TV': 500.0}
}

def build_openai_client():
    if OPENAI_API_KEY is None:
        return None
    if OPENAI_BASE_URL:
        return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    return OpenAI(api_key=OPENAI_API_KEY)

@st.cache_resource(show_spinner=False)
def load_xgb_model(model_path: str = FIXED_MODEL_PATH):
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model not found at {model_path}.")
    booster = xgb.Booster()
    booster.load_model(str(p))
    return booster

def _df_from_patient_input(patient_input: dict) -> pd.DataFrame:
    df = pd.DataFrame([patient_input])
    df = df[REQUIRED_FEATURES]
    return df

# -------------------------------
# 온톨로지 관련 헬퍼
# -------------------------------
def _ontology_label_maps():
    labels = {
        "hemodynamic_instability": "혈역학적 불안정",
        "respiratory_distress": "호흡 곤란",
        "obesity_risk": "비만 위험",
        "age_risk": "고령 위험",
        "airway_obstruction_risk": "기도 폐쇄 위험"
    }
    desc = {
        "hemodynamic_instability": "SpO₂<90% 또는 심박 이상",
        "respiratory_distress": "SpO₂<85%",
        "obesity_risk": "BMI≥30",
        "age_risk": "나이≥65세",
        "airway_obstruction_risk": "비만+호흡곤란 동시"
    }
    return labels, desc

def summarize_ontology_for_report(ontology_json: dict):
    """레포트용: 1(해당)인 온톨로지 항목과 0(비해당) 항목을 구분 정리"""
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
    """온톨로지 결과를 예쁜 테이블(아이콘 포함)로 변환"""
    labels, desc = _ontology_label_maps()
    row = ontology_json["patients"][0]
    rows = []
    for k in labels:
        val = int(row.get(k, 0))
        icon = "✅" if val == 1 else "❌"
        rows.append({"특성": labels[k], "설명": desc[k], "여부": icon})
    return pd.DataFrame(rows)

# -------------------------------
# LLM 태깅 및 룰 기반 Fallback
# -------------------------------
def rule_based_ontology(df: pd.DataFrame) -> dict:
    row = df.iloc[0]
    hemo = int((row.get("SPO2", 100) < 90) or (row.get("HR", 80) < 40) or (row.get("HR", 80) > 120))
    resp = int(row.get("SPO2", 100) < 85)
    ob   = int(row.get("BMI", 0) >= 30)
    age  = int(row.get("AGE", 0) >= 65)
    airway = int(ob and resp)
    return {"patients":[{"patient_index":0,"hemodynamic_instability":hemo,"respiratory_distress":resp,"obesity_risk":ob,"age_risk":age,"airway_obstruction_risk":airway}]}

def llm_tag_ontology(client: OpenAI, df: pd.DataFrame) -> dict:
    patient_records = df.to_dict(orient='records')
    prompt = f"""
당신은 의료 전문가입니다. 다음 환자 데이터를 분석하여 발관(extubation) 시 위험 요인이 될 수 있는 온톨로지 특성을 태깅해주세요.

환자 데이터:
{json.dumps(patient_records, ensure_ascii=False, indent=2)}

다음 특성을 0 또는 1로 반환:
1. hemodynamic_instability
2. respiratory_distress
3. obesity_risk
4. age_risk
5. airway_obstruction_risk

반드시 아래 JSON 스키마만 반환:
{{
  "patients": [{{"patient_index":0,"hemodynamic_instability":0,"respiratory_distress":0,"obesity_risk":0,"age_risk":0,"airway_obstruction_risk":0}}]
}}
"""
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
    feature_names = ["hemodynamic_instability","respiratory_distress","obesity_risk","age_risk","airway_obstruction_risk"]
    for k in feature_names:
        v = ontology_json["patients"][0].get(k, 0)
        df[k] = int(v)
    return df

# -------------------------------
# SHAP 요약 헬퍼
# -------------------------------
def summarize_shap_for_report(shap_exp: dict, top_k: int = 5):
    """
    레포트용: SHAP에서 영향이 큰 상위/하위 요인 리스트를 각각 추립니다.
    상위: |값| 큰 순, sign>0이면 위험↑, sign<0이면 위험↓
    """
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

# -------------------------------
# 예측/SHAP/레포트
# -------------------------------
def run_xgb_predict(booster: xgb.Booster, feature_df: pd.DataFrame):
    dmatrix = xgb.DMatrix(feature_df.values)
    preds = booster.predict(dmatrix)
    p = float(preds[0])
    y = "위험" if p > 0.5 else "안전"   # ← 0/1 → “안전/위험”으로 변경
    return {"probability": p, "class_label": y}

def compute_shap(booster: xgb.Booster, feature_df: pd.DataFrame):
    dmatrix = xgb.DMatrix(feature_df.values)
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(dmatrix)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    sample_vals = shap_values[0]
    names = feature_df.columns.tolist()
    imp_dict = {names[i]: float(sample_vals[i]) for i in range(len(names))}
    top5 = sorted([(k, abs(v)) for k, v in imp_dict.items()], key=lambda x: x[1], reverse=True)[:5]
    return {"shap_values": shap_values, "feature_importance": imp_dict, "top_risk_factors": top5}

def llm_generate_report(client: OpenAI, patient_input: dict, prediction: dict, shap_exp: dict, ontology_json: dict) -> str:
    """
    보호자 설명 레포트를 SHAP + 온톨로지 양쪽 근거로 풍부하게 생성.
    - 예측 클래스(안전/위험)에 따라 톤을 자동 조절
    - 온톨로지에서 1인 항목은 반드시 본문에 반영
    - SHAP 상위 요인은 '왜 그런 예측이 나왔는지'를 설명하는 근거로 사용
    """
    # 온톨로지 요약(1/0 분리)
    ont_pos, ont_neg = summarize_ontology_for_report(ontology_json)
    # SHAP 상·하위 요약
    shap_top, _ = summarize_shap_for_report(shap_exp, top_k=5)

    # 프롬프트에 넣을 JSON/목록(LLM이 정확히 참고하도록)
    ont_pos_for_llm = [{"name": x["name"], "rule": x["rule"]} for x in ont_pos] or [{"name": "해당 항목 없음", "rule": ""}]
    ont_neg_for_llm = [{"name": x["name"], "rule": x["rule"]} for x in ont_neg]
    shap_top_for_llm = shap_top or [{"feature": "해당 없음", "abs_importance": 0.0, "direction": ""}]

    # 클래스에 따른 톤 가이드
    cls_label = prediction.get("class_label", "안전")

    prompt = f"""
당신은 의료진과 환자 보호자 사이의 소통을 돕는 의료 커뮤니케이션 전문가입니다.
아래 '모델 예측', '온톨로지 판단 결과', 'SHAP 중요 요인'을 모두 참고하여,
보호자가 이해하기 쉬운 설명 레포트를 작성하세요.

[모델 예측]
- 발관 실패 확률: {prediction['probability']:.1%}
- 예측 클래스(안전/위험): {cls_label}

[온톨로지 판단 결과]
- 이 환자에게 실제 해당된 항목(값=1): {json.dumps(ont_pos_for_llm, ensure_ascii=False)}
- 해당되지 않은 항목(값=0): {json.dumps(ont_neg_for_llm, ensure_ascii=False)}

[SHAP 중요 요인 Top 5]
- {json.dumps(shap_top_for_llm, ensure_ascii=False)}

[입력 데이터(요약)]
- 일부 주요 수치: {json.dumps({k: patient_input[k] for k in ['AGE','BMI','SPO2','MAP','HR','RR','GCS','PH','PACO2','PAO2','HCO3','LACTATE','FIO2','PEEP','PPLAT','TV'] if k in patient_input}, ensure_ascii=False)}

[작성 지침]
- 톤: 
## 레포트 작성 지침
**목적**: 보호자가 발관(인공호흡기 튜브 제거) 결정을 내리는 데 도움이 되는 정보를 제공합니다.
**톤 & 스타일**:
- 따뜻하고 공감적인 어조를 유지하세요
- 보호자의 불안감을 이해하면서도 정확한 정보를 전달하세요
- 의학적 전문성과 인간미를 균형있게 표현하세요
**용어 선택**:
- 전문 의료 용어는 일반인이 이해하기 쉬운 표현으로 바꿔주세요
  예: "발관/삽관" → "호흡관 제거", "인공호흡기 튜브 제거"
  예: "SPO2" → "혈중 산소포화도" 또는 "산소 수치"
  예: "BMI" → "체질량지수" 또는 "비만도"
- Feature 이름(영어 또는 전문용어)은 한글로 의역하여 설명하세요
**설명 방식**:
- 확률 수치를 구체적인 예시나 비유로 쉽게 설명하세요
- 위험 요인이 왜 중요한지 보호자 입장에서 설명하세요
- 의료진의 모니터링과 전문적 판단이 중요함을 강조하세요
## 레포트 구성 (다음 순서로 작성)
1. **인사 및 소개** (2-3문장)
   - 따뜻한 인사와 레포트의 목적 설명
2. **환자 상태 요약** (3-4문장)
   - 제공된 환자 정보를 일반인이 이해할 수 있는 방식으로 요약
   - 각 수치가 정상 범위인지, 주의가 필요한지 간단히 설명
3. **AI 예측 결과 해석** (4-5문장)
   - 실패 확률이 의미하는 바를 쉽게 설명
   - 이 확률이 높은지 낮은지 맥락 제공
   - 예측이 절대적이 아닌 참고자료임을 명시
4. **주요 위험 요인 상세 설명** (각 요인당 2-3문장)
   - SHAP top 요인 3-5개 위험 요인을 선택하여 설명
   - 각 요인이 왜 발관 성공/실패에 영향을 미치는지 설명
   - Feature 이름을 보호자가 이해할 수 있는 용어로 변환
5. **마무리 메시지** (2문장)
   - 안심과 격려의 메시지
   - 의료진의 전문성에 대한 신뢰 강조
**중요**: 레포트는 순수한 한글 텍스트로만 작성하고, 마크다운 서식(#, **, - 등)은 사용하지 마세요.
섹션 제목은 [섹션명] 형태로 표시하세요.

- 온톨로지에서 값=1인 항목은 '이 환자에게 실제로 관찰된 위험 신호'로 반드시 본문에 포함하세요.
- SHAP 상위 요인은 '왜 이런 예측이 나왔는지' 설명하는 근거로 사용하세요. (증가/감소 방향을 자연스럽게 서술)
- 값=0인 온톨로지 항목은 필요 시 '완화 요인' 또는 '현재는 해당 없음'으로 간단히 언급해도 됩니다.
- 확률 수치는 비유/사례로 쉽게 설명하되, 절대적인 진단이 아님을 분명히 하세요.
- 전문용어는 쉬운 말로 풀어서 설명하세요.
- 마크다운 금지, 섹션 제목은 [섹션명] 형태.
- 분량: 8~14문장 정도.
"""
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
# 세션 상태 초기화
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # [{role, content}]
if "memory" not in st.session_state:
    st.session_state.memory = {}        # 최근 예측 컨텍스트 저장

# -------------------------------
# 메인 UI
# -------------------------------
st.title("Extubation Prediction Demo (Streamlit)")

# (2) 케이스 선택 → 폼 자동 채우기
st.subheader("1) 예시 케이스 선택")
selected_case = st.selectbox("예시 케이스", list(EXAMPLE_CASES.keys()), index=0)
case_vals = EXAMPLE_CASES[selected_case]

# 폼 입력값을 세션에 반영
for k in REQUIRED_FEATURES:
    if f"val_{k}" not in st.session_state:
        st.session_state[f"val_{k}"] = case_vals.get(k, np.nan)

# 케이스 선택이 바뀌면 해당 값으로 초기화
def apply_case(vals: dict):
    for k, v in vals.items():
        st.session_state[f"val_{k}"] = v

if st.button("이 케이스 값 불러오기"):
    apply_case(EXAMPLE_CASES[selected_case])
    st.success(f"{selected_case} 값이 입력 폼에 반영되었습니다.")

# 입력 폼
st.subheader("2) 환자 입력")
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

# 실행 버튼
st.subheader("3) 실행")
run = st.button("예측 실행")

if run:
    # 입력 딕셔너리 만들기
    patient_input = {k: float(st.session_state[f"val_{k}"]) if not np.isnan(st.session_state[f"val_{k}"]) else np.nan
                     for k in REQUIRED_FEATURES}

    # 1) 입력 → DF
    df = _df_from_patient_input(patient_input)

    # 2) 온톨로지 태깅
    with st.spinner("온톨로지 태깅 중..."):
        try:
            if USE_LLM and OPENAI_API_KEY:
                client = build_openai_client()
                ontology_json = llm_tag_ontology(client, df)
            else:
                ontology_json = rule_based_ontology(df)
        except Exception as e:
            st.warning(f"LLM 태깅 실패. 룰 기반으로 대체합니다. ({e})")
            ontology_json = rule_based_ontology(df)

    feature_df = attach_ontology_features(df.copy(), ontology_json)

    # 3) 모델 로드 & 예측
    with st.spinner("XGBoost 예측 중..."):
        try:
            booster = load_xgb_model(FIXED_MODEL_PATH)
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

    # 5) 레포트
    report_text = None
    if USE_LLM and OPENAI_API_KEY:
        with st.spinner("설명 레포트 생성 중..."):
            try:
                client = build_openai_client()
                # ⬇️ 온톨로지 함께 전달
                report_text = llm_generate_report(client, patient_input, pred, shap_exp, ontology_json)
            except Exception as e:
                st.warning(f"레포트 생성 실패: {e}")
                report_text = None

    # 결과 표시
    st.success("완료!")

    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown("### 예측 결과")
        st.metric("발관 실패 확률", f"{pred['probability']*100:.1f}%")
        st.metric("예측 클래스", pred["class_label"])  # “안전/위험”

        st.markdown("### 온톨로지 태깅 결과")
        pretty_df = ontology_pretty_table(ontology_json)
        st.dataframe(pretty_df, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("### 상위 위험 요인 (SHAP | 절대값 Top 5)")
        if "top_risk_factors" in shap_exp and shap_exp["top_risk_factors"]:
            rows = []
            for name, abs_imp in shap_exp["top_risk_factors"]:
                sign = shap_exp["feature_importance"].get(name, 0.0)
                direction = "↑ 위험 증가" if sign > 0 else "↓ 위험 감소"
                rows.append({"feature": name, "abs_importance": round(abs_imp, 5), "direction": direction})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.write("계산되지 않았습니다.")

    # (5) 보호자 설명용 레포트: 편집 가능한 텍스트 박스
    st.markdown("### 보호자 설명용 레포트")

    if report_text:
        # 최초 생성 시 세션 상태에 저장
        if "report_text" not in st.session_state or not st.session_state.get("report_text"):
            st.session_state.report_text = report_text

        # 편집 가능한 텍스트 영역
        st.session_state.report_text = st.text_area(
            "생성된 레포트 (편집 가능)",
            value=st.session_state.report_text,
            height=420,
            help="필요 시 문구를 수정하여 보호자 커뮤니케이션에 활용하세요."
        )

        # 다운로드 버튼 (선택)
        st.download_button(
            label="레포트 .txt 다운로드",
            data=st.session_state.report_text,
            file_name=f"extubation_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    else:
        st.info("레포트가 생성되지 않았습니다. 상단에서 예측을 실행하면 자동으로 생성됩니다.")

    st.markdown("### 모델 입력 전체 Feature (추론 시 사용)")
    st.dataframe(feature_df, use_container_width=True)

    # 채팅 메모리에 저장할 컨텍스트 업데이트
    st.session_state.memory = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "selected_case": selected_case,
        "patient_input": patient_input,
        "prediction": pred,
        "shap_top": shap_exp.get("top_risk_factors", []),
    }

# -------------------------------
# 사이드바: 채팅 (모델/경로 입력 대신)
# -------------------------------
with st.sidebar:
    st.header("💬 환자 보호자를 위한 챗봇 어시스턴트")
    if OPENAI_API_KEY is None:
        st.caption("OpenAI 키가 없어서 채팅은 비활성화됩니다. (Secrets에 OPENAI_API_KEY 추가)")
    else:
        # 최근 예측 컨텍스트를 시스템 메시지로 주입
        context_blob = json.dumps(st.session_state.memory, ensure_ascii=False, indent=2) if st.session_state.memory else "최근 예측 컨텍스트 없음."
        system_msg = (
            "당신은 중환자실에 입실한 환자 보호자를 대하는 의료인입니다. "
            "아래 '최근 예측 컨텍스트'를 참고하여 친절하고 쉽게 답변하세요. 의료인이 아닌 사람들도 알아들을 수 있도록 설명하세요. \n\n"
            f"[최근 예측 컨텍스트]\n{context_blob}"
        )

        # 기존 대화 표시
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
                messages = [{"role":"system","content":system_msg}] + st.session_state.chat_history[-20:]  # 최근 20턴
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