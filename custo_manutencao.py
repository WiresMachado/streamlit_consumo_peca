import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(
    page_title="Manutenção / Consumo de Peças",
    layout="wide"
)

# ------------------------------------------------------------
# Funções auxiliares de formatação/normalização
# ------------------------------------------------------------

def init_session_state():
    defaults = {
        "df_pecas_raw": None,
        "df_custos_raw": None,
        "df_maquinas_raw": None,

        "modelo_selecionado": None,
        "chassi_selecionado": None,

        "hectare_ano_ref": None,
        "hectare_hora_ref": None,
        "largura_ref_m": None,
        "modo_operacao": "Moderado",

        "prod_base": "Por máquina",

        "df_maquinas_proc": None,
        "resumo_maquina_ref": None,
        "df_pecas_proc": None,

        "filtro_campo": "Todos",
        "filtro_valor": "",
        "filtro_familia": "Todos",

        "filtro_familia_resumo": "Todos",
        "escopo_resumo": "Apenas chassi selecionado",

        # >>> novos estados de filtro da página 3 <<<
        "filtro_campo_resumo": "Todos",
        "filtro_valor_resumo": "",

        # Persistência dos ajustes por Código
        # {"00001234": {"hect": float|None, "prop": int|None,
        #               "manual_hect": bool, "manual_prop": bool}}
        "ajustes_pecas": {},

        # Assinatura do processamento para evitar reconstrução desnecessária
        "assinatura_processamento": None,

        # Parâmetros globais ajustáveis
        "default_proporcao_troca": 50,
        "multiplicadores_operacao": {"Leve": 1.5, "Moderado": 1.0, "Extremo": 0.6},

        # Estado para importação de ajustes
        "ajustes_import_df": None,
        "ajustes_import_filename": None,
        "ajustes_import_applied": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def aplicar_modo_operacao(valor_hectare_prop, modo):
    mults = st.session_state.get("multiplicadores_operacao",
                                 {"Leve":1.5,"Moderado":1.0,"Extremo":0.6})
    mult = float(mults.get(modo, 1.0))
    try:
        return float(valor_hectare_prop) * mult
    except:
        return 0.0


def format_currency(v):
    if pd.isna(v):
        return "R$ 0,00"
    try:
        return "R$ " + f"{float(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return "R$ 0,00"


def _strip_all(s):
    return str(s).replace(".", "").replace(",", "").replace("-", "").replace(" ", "")


def format_codigo(cod):
    if pd.isna(cod):
        return "00000000"
    return _strip_all(cod).zfill(8)


def format_ano(ano):
    if pd.isna(ano):
        return ""
    s = str(int(float(ano))) if str(ano).replace(".", "", 1).isdigit() else str(ano)
    return s.zfill(4)


def format_thousand_no_decimals(v):
    if pd.isna(v):
        return "0"
    try:
        inteiro = int(round(float(v)))
    except:
        return "0"
    return f"{inteiro:,}".replace(",", ".")


def format_hectare_original(v):
    return format_thousand_no_decimals(v)


# ------------------------------------------------------------
# HIGIENIZAÇÕES
# ------------------------------------------------------------

def higienizar_pecas(df_pecas):
    df = df_pecas.copy()
    if "Código" in df.columns:
        df["Código"] = df["Código"].apply(format_codigo)
    return df


def higienizar_custos(df_custos):
    df = df_custos.copy()
    if "Código" in df.columns:
        df["Código"] = df["Código"].apply(format_codigo)
    else:
        raise ValueError("Tabela Custos precisa ter a coluna 'Código'.")
    if "Custo" not in df.columns:
        raise ValueError("Tabela Custos precisa ter a coluna 'Custo'.")
    df["Custo"] = pd.to_numeric(df["Custo"], errors="coerce")
    df = df.dropna(subset=["Código", "Custo"])
    df = df.sort_index().drop_duplicates(subset=["Código"], keep="last").reset_index(drop=True)
    return df


def higienizar_maquinas(df_maquinas):
    dfm = df_maquinas.copy()
    for col in ["Modelo", "Chassi"]:
        if col in dfm.columns:
            dfm[col] = dfm[col].astype(str).str.strip()
    for col in ["Linhas", "Espaçamento", "Ano"]:
        if col in dfm.columns:
            dfm[col] = pd.to_numeric(dfm[col], errors="coerce")
    if set(["Modelo", "Chassi"]).issubset(dfm.columns):
        dfm = dfm.drop_duplicates(subset=["Modelo", "Chassi"], keep="first")
    return dfm


# ------------------------------------------------------------
# Processamento (produtividade e cálculo de consumo)
# ------------------------------------------------------------

def processar_maquinas(
    df_maquinas, hectare_ano_ref, hectare_hora_ref, largura_ref_m,
    modelo_escolhido, chassi_ref_escolhido, prod_base
):
    df = df_maquinas.copy()
    df = df[df["Modelo"] == modelo_escolhido].copy()

    if df.empty:
        resumo_ref = {
            "chassi_ref": chassi_ref_escolhido if chassi_ref_escolhido else None,
            "linhas_maquina": 0,
            "ha_ano_maquina": 0.0,
            "ha_hora_maquina": 0.0,
            "horas_maquina_ano": 0.0,
            "n_chassis_frota": 0,
            "modelo": modelo_escolhido,
        }
        return pd.DataFrame(), resumo_ref

    df["largura_total_m"] = (df["Linhas"] * df["Espaçamento"]) / 100.0

    if prod_base == "Por máquina":
        df["ha_ano_chassi"] = float(hectare_ano_ref or 0.0)
        df["ha_hora_chassi"] = float(hectare_hora_ref or 0.0)
        df["horas_chassi_ano"] = np.where(
            df["ha_hora_chassi"] > 0,
            df["ha_ano_chassi"] / df["ha_hora_chassi"],
            np.nan
        )
    else:
        if largura_ref_m and largura_ref_m != 0:
            ha_hora_por_metro_ref = float(hectare_hora_ref or 0.0) / largura_ref_m
            ha_ano_por_metro_ref  = float(hectare_ano_ref  or 0.0) / largura_ref_m
        else:
            ha_hora_por_metro_ref = np.nan
            ha_ano_por_metro_ref  = np.nan

        df["ha_hora_chassi"] = df["largura_total_m"] * ha_hora_por_metro_ref
        df["ha_ano_chassi"]  = df["largura_total_m"] * ha_ano_por_metro_ref
        df["horas_chassi_ano"] = df["ha_ano_chassi"] / df["ha_hora_chassi"]

    df_sorted = df.sort_values(by="Chassi").reset_index(drop=True)
    chassis_lista = df_sorted["Chassi"].astype(str).tolist()

    if chassi_ref_escolhido == "Todos":
        usar_chassi_para_calculo = chassis_lista[0]
    else:
        usar_chassi_para_calculo = chassi_ref_escolhido

    if usar_chassi_para_calculo not in chassis_lista:
        usar_chassi_para_calculo = chassis_lista[0]

    linha_ref = df_sorted[df_sorted["Chassi"].astype(str) == str(usar_chassi_para_calculo)].iloc[0]
    linhas_ref = int(linha_ref["Linhas"])

    resumo_ref = {
        "chassi_ref": chassi_ref_escolhido,
        "linhas_maquina": linhas_ref,
        "ha_ano_maquina": float(linha_ref["ha_ano_chassi"]),
        "ha_hora_maquina": float(linha_ref["ha_hora_chassi"]),
        "horas_maquina_ano": float(linha_ref["horas_chassi_ano"]) if pd.notna(linha_ref["horas_chassi_ano"]) else 0.0,
        "n_chassis_frota": int(len(df_sorted)),
        "modelo": modelo_escolhido,
    }

    return df_sorted, resumo_ref


def _quantidade_recomendada_uma_maquina(row, resumo_maquina_ref):
    try:
        vida_base = float(row["hectare_proporcao_efetivo"])
        qtd_por_prop = float(row["Qtd/Proporção"])
        prop_troca = float(row["proporcao_troca_%"])
        tipo_prop = str(row["Proporção"]).strip().lower()
    except Exception:
        return 0.0

    if vida_base <= 0:
        return 0.0

    ha_ano = float(resumo_maquina_ref.get("ha_ano_maquina", 0.0) or 0.0)
    n_linhas = int(resumo_maquina_ref.get("linhas_maquina", 1) or 1)

    if tipo_prop == "linha":
        vida_total = vida_base * n_linhas
        qtd_total_por_ciclo = qtd_por_prop * n_linhas
    else:
        vida_total = vida_base
        qtd_total_por_ciclo = qtd_por_prop

    if vida_total <= 0:
        return 0.0

    ciclos = ha_ano / vida_total
    consumo_teorico = ciclos * qtd_total_por_ciclo
    qtd_rec = consumo_teorico * (prop_troca / 100.0)
    return qtd_rec


# ---------------- REAPLICAÇÃO DE AJUSTES (respeita apenas o que foi MANUAL) ----------------

def _reaplicar_ajustes(df):
    """
    Reaplica somente os campos ajustados MANUALMENTE em st.session_state['ajustes_pecas'].
    Se não for manual, mantém o valor recalculado (permitindo que o modo/multiplicador atualizem a base).
    """
    ajustes = st.session_state.get("ajustes_pecas", {})
    if not isinstance(ajustes, dict) or df.empty:
        return df

    df["Código"] = df["Código"].apply(format_codigo)

    for cod, vals in ajustes.items():
        cod_norm = format_codigo(cod)
        m = df["Código"] == cod_norm
        if not m.any() or not isinstance(vals, dict):
            continue

        if vals.get("manual_hect", False) and ("hect" in vals) and (vals["hect"] is not None):
            df.loc[m, "hectare_proporcao_efetivo"] = float(vals["hect"])

        if vals.get("manual_prop", False) and ("prop" in vals) and (vals["prop"] is not None):
            df.loc[m, "proporcao_troca_%"] = int(vals["prop"])

    return df


def construir_df_pecas(df_pecas, df_custos, resumo_maquina_ref, modo_operacao):
    if df_pecas is None or df_custos is None or df_pecas.empty:
        return pd.DataFrame()

    dfp = higienizar_pecas(df_pecas)
    dfc = higienizar_custos(df_custos)

    modelo_sel = st.session_state.get("modelo_selecionado")
    if "Modelo" in dfp.columns and modelo_sel:
        dfp = dfp[dfp["Modelo"] == modelo_sel].copy()

    if "Código" in dfp.columns:
        dfp = dfp.drop_duplicates(subset=["Código"], keep="first")

    df = dfp.merge(dfc[["Código", "Custo"]], on="Código", how="left")

    # Base inicial (afetada por modo e multiplicadores)
    df["hectare_proporcao_efetivo"] = df["Hectare/Proporção"].apply(
        lambda x: aplicar_modo_operacao(x, st.session_state["modo_operacao"])
    )

    # Proporção padrão global (pode ser mudada manualmente por item na página 2)
    df["proporcao_troca_%"] = float(st.session_state.get("default_proporcao_troca", 50))

    df["custo_unitario"] = df["Custo"]
    df["custo_total_base"] = df["Qtd/Proporção"] * df["custo_unitario"]

    # Reaplica somente os ajustes manuais
    df = _reaplicar_ajustes(df)

    # Recalcula quantidade e custo planejado (por máquina ref)
    df["qtd_recomendada"] = df.apply(
        lambda r: _quantidade_recomendada_uma_maquina(r, resumo_maquina_ref),
        axis=1
    )
    df["custo_planejado_item"] = df["qtd_recomendada"] * df["custo_unitario"]

    dedup_cols = ["Código","Descrição","Família","Proporção","Qtd/Proporção",
                  "hectare_proporcao_efetivo","proporcao_troca_%","custo_unitario"]
    dedup_cols = [c for c in dedup_cols if c in df.columns]
    df = df.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)

    return df


def recalcular_pecas_pos_ajuste(df_pecas_proc, resumo_maquina_ref):
    if df_pecas_proc is None or df_pecas_proc.empty:
        return df_pecas_proc
    df = df_pecas_proc.copy()

    # Reaplica somente os ajustes manuais (garantia extra)
    df = _reaplicar_ajustes(df)

    dedup_cols = ["Código","Descrição","Família","Proporção","Qtd/Proporção",
                  "hectare_proporcao_efetivo","proporcao_troca_%","custo_unitario"]
    dedup_cols = [c for c in dedup_cols if c in df.columns]
    df = df.drop_duplicates(subset=dedup_cols, keep="first")

    df["qtd_recomendada"] = df.apply(
        lambda r: _quantidade_recomendada_uma_maquina(r, resumo_maquina_ref),
        axis=1
    )
    df["custo_planejado_item"] = df["qtd_recomendada"] * df["custo_unitario"]
    return df


def agregar_para_exportacao(df_pecas_proc, resumo_maquina_ref, familia_filter="Todos", escopo="Apenas chassi selecionado"):
    if df_pecas_proc is None or df_pecas_proc.empty:
        return pd.DataFrame(columns=["Código","Descrição","Família","Qtd recomendada","Custo total"])

    n_chassis_total = int(resumo_maquina_ref.get("n_chassis_frota", 1) or 1)
    n_chassis = 1 if escopo == "Apenas chassi selecionado" else n_chassis_total

    df_escalada = df_pecas_proc.copy()
    if familia_filter != "Todos":
        df_escalada = df_escalada[df_escalada["Família"] == familia_filter]

    cols_dedup = ["Código","Descrição","Família","Proporção","Qtd/Proporção",
                  "hectare_proporcao_efetivo","proporcao_troca_%","custo_unitario"]
    cols_dedup = [c for c in cols_dedup if c in df_escalada.columns]
    df_escalada = df_escalada.drop_duplicates(subset=cols_dedup, keep="first")

    df_escalada["Qtd recomendada (escopo)"] = df_escalada["qtd_recomendada"] * n_chassis
    df_escalada["Custo total (escopo)"]     = df_escalada["custo_planejado_item"] * n_chassis

    agr = (
        df_escalada
        .groupby("Código", as_index=False, sort=False)
        .agg({
            "Descrição": "first",
            "Família": "first",
            "Qtd recomendada (escopo)": "sum",
            "Custo total (escopo)": "sum"
        })
        .rename(columns={
            "Qtd recomendada (escopo)": "Qtd recomendada",
            "Custo total (escopo)":     "Custo total"
        })
    )
    agr = agr.sort_values(by="Código").reset_index(drop=True)
    return agr


def gerar_planilha_exportacao(df_pecas_proc, resumo_maquina_ref, familia_filter="Todos", escopo="Apenas chassi selecionado"):
    df_export = agregar_para_exportacao(df_pecas_proc, resumo_maquina_ref, familia_filter, escopo)
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df_export.to_excel(writer, index=False, sheet_name="Planejado")
    buffer.seek(0)
    return buffer


def calcular_indicadores_resumo(df_pecas_proc, resumo_maquina_ref, escopo="Apenas chassi selecionado"):
    if df_pecas_proc is None or df_pecas_proc.empty:
        return {
            "custo_total_estoque": 0.0,
            "custo_medio_por_hectare": 0.0,
            "custo_medio_por_hora": 0.0
        }

    custo_total_por_maquina = df_pecas_proc["custo_planejado_item"].sum()
    n_chassis_total = int(resumo_maquina_ref.get("n_chassis_frota", 1) or 1)
    n_chassis = 1 if escopo == "Apenas chassi selecionado" else n_chassis_total
    custo_total_escopo = custo_total_por_maquina * n_chassis

    ha_ano    = resumo_maquina_ref.get("ha_ano_maquina", 0.0)
    horas_ano = resumo_maquina_ref.get("horas_maquina_ano", 0.0)

    custo_medio_por_hectare = (custo_total_por_maquina / ha_ano) if ha_ano else np.nan
    custo_medio_por_hora    = (custo_total_por_maquina / horas_ano) if horas_ano else np.nan

    return {
        "custo_total_estoque": custo_total_escopo,
        "custo_medio_por_hectare": custo_medio_por_hectare,
        "custo_medio_por_hora": custo_medio_por_hora
    }


# ----------------- AUDITORIA -----------------

def auditar_item(row_item, resumo_maquina_ref):
    try:
        vida_base = float(row_item["hectare_proporcao_efetivo"])
        qtd_por_prop = float(row_item["Qtd/Proporção"])
        prop_troca = float(row_item["proporcao_troca_%"])
        tipo_prop = str(row_item["Proporção"]).strip().lower()
    except Exception:
        return {}

    ha_ano = float(resumo_maquina_ref.get("ha_ano_maquina", 0.0) or 0.0)
    n_linhas = int(resumo_maquina_ref.get("linhas_maquina", 1) or 1)

    if tipo_prop == "linha":
        vida_total = vida_base * n_linhas
        qtd_total_por_ciclo = qtd_por_prop * n_linhas
    else:
        vida_total = vida_base
        qtd_total_por_ciclo = qtd_por_prop

    ciclos = ha_ano / vida_total if vida_total > 0 else 0
    consumo = ciclos * qtd_total_por_ciclo
    qtd_final = consumo * (prop_troca / 100.0)

    return {
        "n_linhas": n_linhas,
        "ha_ano_maquina": ha_ano,
        "vida_por_linha_ou_maq": vida_base,
        "vida_total": vida_total,
        "qtd_por_linha_ou_maq": qtd_por_prop,
        "qtd_total_por_ciclo": qtd_total_por_ciclo,
        "ciclos_ano": ciclos,
        "proporcao_troca_%": prop_troca,
        "qtd_final": qtd_final
    }


# =================== Cálculo auxiliar p/ Página 2 ===================

def calcular_hect_ref_e_qtd_prevista(row, resumo_maquina_ref, hectare_efetivo_atual, proporcao_troca_atual):
    """
    Calcula:
      - Hectare referência (vida_total): se Proporção='Linha' => hectare_efetivo * n_linhas; senão, por máquina.
      - Quantidade prevista: usa ha_ano do chassi selecionado, vida_total e Qtd/Proporção,
        aplicando a proporção de troca informada no input atual.
    """
    try:
        tipo_prop = str(row["Proporção"]).strip().lower()
        n_linhas = int(resumo_maquina_ref.get("linhas_maquina", 1) or 1)
        qtd_por_prop = float(row["Qtd/Proporção"])
        ha_ano = float(resumo_maquina_ref.get("ha_ano_maquina", 0.0) or 0.0)
        vida_base = float(hectare_efetivo_atual)
        prop_troca = float(proporcao_troca_atual)
    except Exception:
        return 0.0, 0.0

    if tipo_prop == "linha":
        vida_total = vida_base * n_linhas
        qtd_total_por_ciclo = qtd_por_prop * n_linhas
    else:
        vida_total = vida_base
        qtd_total_por_ciclo = qtd_por_prop

    if vida_total > 0:
        ciclos = ha_ano / vida_total
        consumo_teorico = ciclos * qtd_total_por_ciclo
        qtd_prevista = consumo_teorico * (prop_troca / 100.0)
    else:
        qtd_prevista = 0.0

    return float(vida_total), float(qtd_prevista)


# ------------------------------------------------------------
# === AJUSTES: EXPORT/IMPORT ===
# ------------------------------------------------------------

def montar_df_ajustes_atual():
    """
    Retorna DataFrame com:
      Código, Hectare/Proporção, Proporção de troca (%)
    usando os valores ATUAIS da página 2 (já com ajustes/modo).
    """
    df = st.session_state.get("df_pecas_proc")
    if df is None or df.empty:
        return pd.DataFrame(columns=["Código", "Hectare/Proporção", "Proporção de troca (%)"])
    tmp = df.copy()
    tmp["Código"] = tmp["Código"].apply(format_codigo)
    base = (
        tmp.groupby("Código", as_index=False)
        .agg({
            "hectare_proporcao_efetivo": "first",
            "proporcao_troca_%": "first"
        })
        .rename(columns={
            "hectare_proporcao_efetivo": "Hectare/Proporção",
            "proporcao_troca_%": "Proporção de troca (%)"
        })
        .sort_values("Código")
        .reset_index(drop=True)
    )
    return base


def gerar_planilha_ajustes():
    """
    Gera buffer Excel com os ajustes atuais (todos os códigos, inclusive sem ajustes manuais).
    """
    df_exp = montar_df_ajustes_atual()
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df_exp.to_excel(writer, index=False, sheet_name="Ajustes")
    buffer.seek(0)
    return buffer


def aplicar_importacao_ajustes(df_import):
    """
    Aplica em massa os ajustes vindos do Excel (Código, Hectare/Proporção, Proporção de troca (%)).
    Marca como MANUAL e recalcula base da página 2.
    """
    if df_import is None or df_import.empty:
        return False, "Arquivo vazio ou inválido."

    # Mapeia colunas por case-insensitive
    cols = {c.strip().lower(): c for c in df_import.columns}
    req = {
        "código": None,
        "hectare/proporção": None,
        "proporção de troca (%)": None
    }
    for k in list(req.keys()):
        if k in cols:
            req[k] = cols[k]
    if None in req.values():
        return False, "As colunas obrigatórias são: Código, Hectare/Proporção, Proporção de troca (%)."

    df = df_import[[req["código"], req["hectare/proporção"], req["proporção de troca (%)"]]].copy()
    df.columns = ["Código", "Hectare/Proporção", "Proporção de troca (%)"]

    # Normaliza e coerce
    df["Código"] = df["Código"].apply(format_codigo)
    df["Hectare/Proporção"] = pd.to_numeric(df["Hectare/Proporção"], errors="coerce").fillna(0.0)
    df["Proporção de troca (%)"] = pd.to_numeric(df["Proporção de troca (%)"], errors="coerce").fillna(0.0).astype(int)

    ajustes = st.session_state.get("ajustes_pecas", {}).copy()

    # Aplica como manual
    for _, r in df.iterrows():
        cod = r["Código"]
        ajustes[cod] = {
            "hect": float(r["Hectare/Proporção"]),
            "prop": int(r["Proporção de troca (%)"]),
            "manual_hect": True,
            "manual_prop": True,
        }

    st.session_state["ajustes_pecas"] = ajustes

    # Reflete na df_pecas_proc atual e recalcula
    if st.session_state.get("df_pecas_proc") is not None and not st.session_state["df_pecas_proc"].empty:
        st.session_state["df_pecas_proc"]["Código"] = st.session_state["df_pecas_proc"]["Código"].apply(format_codigo)
        for _, r in df.iterrows():
            m = st.session_state["df_pecas_proc"]["Código"] == r["Código"]
            st.session_state["df_pecas_proc"].loc[m, "hectare_proporcao_efetivo"] = float(r["Hectare/Proporção"])
            st.session_state["df_pecas_proc"].loc[m, "proporcao_troca_%"] = int(r["Proporção de troca (%)"])

        st.session_state["df_pecas_proc"] = recalcular_pecas_pos_ajuste(
            st.session_state["df_pecas_proc"],
            st.session_state["resumo_maquina_ref"]
        )

    return True, "Importação aplicada com sucesso."


# ------------------------------------------------------------
# Assinatura (para evitar reset ao navegar) + Reprocessamento central
# ------------------------------------------------------------

def _assinatura_atual():
    """Cria uma tupla hashable com tudo que influencia o processamento."""
    mults = st.session_state.get("multiplicadores_operacao",
                                 {"Leve":1.5,"Moderado":1.0,"Extremo":0.6})
    return (
        id(st.session_state.get("df_pecas_raw")),
        id(st.session_state.get("df_custos_raw")),
        id(st.session_state.get("df_maquinas_raw")),
        st.session_state.get("modelo_selecionado"),
        st.session_state.get("chassi_selecionado"),
        st.session_state.get("hectare_ano_ref"),
        st.session_state.get("hectare_hora_ref"),
        st.session_state.get("largura_ref_m"),
        st.session_state.get("modo_operacao"),
        st.session_state.get("prod_base"),
        float(mults.get("Leve",1.5)),
        float(mults.get("Moderado",1.0)),
        float(mults.get("Extremo",0.6)),
        int(st.session_state.get("default_proporcao_troca",50)),
    )

def _pode_processar():
    return all([
        st.session_state["df_pecas_raw"] is not None,
        st.session_state["df_custos_raw"] is not None,
        st.session_state["df_maquinas_raw"] is not None,
        st.session_state["modelo_selecionado"] is not None,
        st.session_state["chassi_selecionado"] is not None
    ])

def run_processamento_if_needed(show_msg=False):
    """
    Reprocessa máquinas e peças quando a assinatura mudar.
    É chamada em TODAS as páginas, garantindo que alterações no modo/multiplicadores
    reflitam imediatamente na página 2/3.
    """
    if not _pode_processar():
        return

    nova_assinatura = _assinatura_atual()
    assinatura_antiga = st.session_state.get("assinatura_processamento")

    if (st.session_state["df_pecas_proc"] is None) or (nova_assinatura != assinatura_antiga):
        df_maquinas_proc, resumo_ref = processar_maquinas(
            st.session_state["df_maquinas_raw"],
            st.session_state["hectare_ano_ref"],
            st.session_state["hectare_hora_ref"],
            st.session_state["largura_ref_m"],
            st.session_state["modelo_selecionado"],
            st.session_state["chassi_selecionado"],
            st.session_state["prod_base"]
        )
        st.session_state["df_maquinas_proc"] = df_maquinas_proc
        st.session_state["resumo_maquina_ref"] = resumo_ref

        st.session_state["df_pecas_proc"] = construir_df_pecas(
            st.session_state["df_pecas_raw"],
            st.session_state["df_custos_raw"],
            resumo_ref,
            st.session_state["modo_operacao"]
        )

        st.session_state["assinatura_processamento"] = nova_assinatura
        if show_msg:
            st.success("Dados reprocessados com base nos parâmetros atuais.")
    else:
        if show_msg:
            st.info("Parâmetros não mudaram. Mantendo cálculos e ajustes atuais.")


# ------------------------------------------------------------
# Layout principal (páginas)
# ------------------------------------------------------------

init_session_state()
pagina = st.sidebar.radio(
    "Navegação",
    ["1. Entrada de Dados", "2. Ajustes de Peças", "3. Resumo / Resultados"]
)

# ------------------------------------------------------------
# PÁGINA 1 - Entrada de Dados
# ------------------------------------------------------------
if pagina == "1. Entrada de Dados":
    st.title("1. Entrada de Dados")

    st.subheader("Importar planilhas (.xlsx)")
    col_up1, col_up2, col_up3 = st.columns(3)
    with col_up1:
        pecas_file = st.file_uploader("Tabela Peças", type=["xlsx"], key="upload_pecas")
    with col_up2:
        custos_file = st.file_uploader("Tabela Custos", type=["xlsx"], key="upload_custos")
    with col_up3:
        maquinas_file = st.file_uploader("Tabela Máquinas", type=["xlsx"], key="upload_maquinas")

    if pecas_file is not None:
        st.session_state["df_pecas_raw"] = pd.read_excel(pecas_file)

    if custos_file is not None:
        st.session_state["df_custos_raw"] = pd.read_excel(custos_file)

    if maquinas_file is not None:
        st.session_state["df_maquinas_raw"] = higienizar_maquinas(pd.read_excel(maquinas_file))

    with st.expander("Pré-visualizar dados importados"):
        if st.session_state["df_pecas_raw"] is not None:
            prev_pecas = higienizar_pecas(st.session_state["df_pecas_raw"]).copy()
            if "Hectare/Proporção" in prev_pecas.columns:
                prev_pecas["Hectare/Proporção"] = prev_pecas["Hectare/Proporção"].apply(
                    format_thousand_no_decimals
                )
            st.write("Peças (formatado p/ visualização):", prev_pecas)

        if st.session_state["df_custos_raw"] is not None:
            prev_custos = higienizar_custos(st.session_state["df_custos_raw"]).copy()
            if "Custo" in prev_custos.columns:
                prev_custos["Custo"] = prev_custos["Custo"].apply(format_currency)
            st.write("Custos (formatado p/ visualização):", prev_custos)

        if st.session_state["df_maquinas_raw"] is not None:
            prev_maqs = st.session_state["df_maquinas_raw"].copy()
            if "Chassi" in prev_maqs.columns:
                prev_maqs["Chassi"] = prev_maqs["Chassi"].apply(format_codigo)
            if "Ano" in prev_maqs.columns:
                prev_maqs["Ano"] = prev_maqs["Ano"].apply(format_ano)
            st.write("Máquinas (formatado p/ visualização):", prev_maqs)

    st.markdown("---")
    st.subheader("Parâmetros operacionais")

    st.session_state["prod_base"] = st.radio(
        "Base do Hectare médio por ano informado:",
        ["Por máquina", "Por metro de largura"],
        horizontal=True,
        index=(0 if st.session_state["prod_base"] == "Por máquina" else 1)
    )

    if st.session_state["df_maquinas_raw"] is not None:
        modelos_disponiveis = sorted(st.session_state["df_maquinas_raw"]["Modelo"].dropna().unique())
    else:
        modelos_disponiveis = []

    col_in1, col_in2 = st.columns(2)

    with col_in1:
        st.session_state["modelo_selecionado"] = st.selectbox(
            "Modelo da máquina",
            modelos_disponiveis,
            index=(
                modelos_disponiveis.index(st.session_state["modelo_selecionado"])
                if st.session_state["modelo_selecionado"] in modelos_disponiveis
                else 0 if modelos_disponiveis else None
            )
        )

        st.session_state["hectare_ano_ref"] = st.number_input(
            "Hectare médio por ano",
            min_value=0,
            step=1,
            value=st.session_state["hectare_ano_ref"] if st.session_state["hectare_ano_ref"] else 0
        )

        st.session_state["hectare_hora_ref"] = st.number_input(
            "Hectares por hora",
            min_value=0.0,
            step=0.1,
            value=st.session_state["hectare_hora_ref"] if st.session_state["hectare_hora_ref"] else 0.0
        )

    with col_in2:
        st.session_state["largura_ref_m"] = st.number_input(
            "Largura do equipamento (m)",
            min_value=0.0,
            step=0.1,
            value=st.session_state["largura_ref_m"] if st.session_state["largura_ref_m"] else 0.0
        )

        st.write("Modo de operação:")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Leve"):
                st.session_state["modo_operacao"] = "Leve"
        with c2:
            if st.button("Moderado"):
                st.session_state["modo_operacao"] = "Moderado"
        with c3:
            if st.button("Extremo"):
                st.session_state["modo_operacao"] = "Extremo"
        st.info(f"Modo atual: {st.session_state['modo_operacao']}")

    # Expander: parâmetros avançados (multiplicadores + proporção padrão)
    with st.expander("Parâmetros avançados (opcional)"):
        st.session_state["modo_operacao"] = st.selectbox(
            "Modo de operação",
            ["Leve", "Moderado", "Extremo"],
            index=["Leve", "Moderado", "Extremo"].index(st.session_state["modo_operacao"])
        )

        mults = st.session_state["multiplicadores_operacao"]
        cA, cB, cC = st.columns(3)
        with cA:
            mults["Leve"] = st.number_input("Multiplicador - Leve",
                                            min_value=0.1, max_value=5.0, step=0.1,
                                            value=float(mults.get("Leve",1.5)))
        with cB:
            mults["Moderado"] = st.number_input("Multiplicador - Moderado",
                                                min_value=0.1, max_value=5.0, step=0.1,
                                                value=float(mults.get("Moderado",1.0)))
        with cC:
            mults["Extremo"] = st.number_input("Multiplicador - Extremo",
                                               min_value=0.1, max_value=5.0, step=0.1,
                                               value=float(mults.get("Extremo",0.6)))
        st.session_state["multiplicadores_operacao"] = mults
        st.caption("Os multiplicadores acima são aplicados sobre o Hectare/Proporção de cada peça.")

        st.session_state["default_proporcao_troca"] = st.slider(
            "Proporção de troca padrão (%)",
            min_value=0, max_value=100, step=1,
            value=int(st.session_state["default_proporcao_troca"])
        )
        st.caption("Esse valor inicial pode ser alterado peça a peça na página 2.")

    # Chassi
    chassi_opcoes = []
    if (
        st.session_state["df_maquinas_raw"] is not None and
        st.session_state["modelo_selecionado"] is not None
    ):
        mask_modelo = st.session_state["df_maquinas_raw"]["Modelo"] == st.session_state["modelo_selecionado"]
        chassi_opcoes = (
            st.session_state["df_maquinas_raw"]
            .loc[mask_modelo, "Chassi"]
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        )
    chassi_dropdown = ["Todos"] + chassi_opcoes if chassi_opcoes else []

    st.session_state["chassi_selecionado"] = st.selectbox(
        "Chassi",
        chassi_dropdown,
        index=(
            chassi_dropdown.index(st.session_state["chassi_selecionado"])
            if st.session_state["chassi_selecionado"] in chassi_dropdown
            else 0 if chassi_dropdown else None
        )
    )

    st.markdown("---")

    # Reprocessamento central também na página 1
    run_processamento_if_needed(show_msg=True)


# ------------------------------------------------------------
# PÁGINA 2 - Ajustes de Peças (com callbacks para sincronização + export/import)
# ------------------------------------------------------------
elif pagina == "2. Ajustes de Peças":
    st.title("2. Ajustes de Peças")

    # Reprocessa aqui também
    run_processamento_if_needed(show_msg=False)

    if (
        st.session_state["df_pecas_proc"] is None
        or st.session_state["df_maquinas_proc"] is None
        or st.session_state["df_pecas_proc"].empty
        or st.session_state["resumo_maquina_ref"] is None
    ):
        st.warning("Primeiro importe os dados e processe na página '1. Entrada de Dados'.")
    else:
        st.write("Edite os parâmetros peça a peça. Esses ajustes alimentam os cálculos finais.")
        st.write("Os valores **permanecem salvos** ao alternar páginas; só se perdem ao recarregar o app ou importar novas tabelas.")

        # ====== EXPORTAR/IMPORTAR AJUSTES ======
        with st.expander("Exportar / Importar ajustes (backup)"):
            c1, c2 = st.columns([1, 2])
            with c1:
                buffer_aj = gerar_planilha_ajustes()
                st.download_button(
                    label="Exportar ajustes (Excel)",
                    data=buffer_aj,
                    file_name="ajustes_pecas.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            with c2:
                up_file = st.file_uploader(
                    "Importar ajustes (.xlsx) com colunas: Código, Hectare/Proporção, Proporção de troca (%)",
                    type=["xlsx"],
                    key="upload_ajustes_xlsx"
                )

                # Se um arquivo foi enviado, carregamos e guardamos em session_state
                if up_file is not None:
                    # Se for um novo arquivo (nome mudou), recarrega para o estado
                    if st.session_state["ajustes_import_filename"] != up_file.name:
                        try:
                            st.session_state["ajustes_import_df"] = pd.read_excel(up_file)
                            st.session_state["ajustes_import_filename"] = up_file.name
                            st.session_state["ajustes_import_applied"] = False
                        except Exception as e:
                            st.error(f"Falha ao ler o arquivo: {e}")
                            st.session_state["ajustes_import_df"] = None

                df_imp = st.session_state.get("ajustes_import_df")

                if df_imp is not None:
                    st.caption("Pré-visualização dos ajustes importados:")
                    st.dataframe(df_imp.head(), use_container_width=True)

                    if st.button("Aplicar ajustes importados", use_container_width=True):
                        ok, msg = aplicar_importacao_ajustes(df_imp)
                        if ok:
                            st.session_state["ajustes_import_applied"] = True
                            st.success(msg)
                            # Força um rerun apenas uma vez para recarregar os widgets com os novos valores
                            st.rerun()
                        else:
                            st.error(msg)
                else:
                    st.caption("Nenhum arquivo de ajustes foi carregado ainda.")

        df_full = st.session_state["df_pecas_proc"].copy()

        familias_distintas = sorted(df_full["Família"].dropna().unique().tolist())
        familias_dropdown = ["Todos"] + familias_distintas

        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1.2, 1.2, 2])
        with col_ctrl1:
            st.session_state["filtro_familia"] = st.selectbox(
                "Família",
                familias_dropdown,
                index=(
                    familias_dropdown.index(st.session_state["filtro_familia"])
                    if st.session_state["filtro_familia"] in familias_dropdown
                    else 0
                )
            )

        with col_ctrl2:
            st.session_state["filtro_campo"] = st.selectbox(
                "Filtrar por campo",
                ["Todos", "Código", "Descrição", "Família"],
                index=["Todos", "Código", "Descrição", "Família"].index(st.session_state["filtro_campo"])
            )

        with col_ctrl3:
            st.session_state["filtro_valor"] = st.text_input(
                "Valor do filtro (contém)",
                value=st.session_state["filtro_valor"]
            )

        df_unique = df_full.groupby("Código").first().reset_index()

        fam_sel = st.session_state["filtro_familia"]
        if fam_sel != "Todos":
            df_unique = df_unique[df_unique["Família"] == fam_sel]

        filtro_txt = st.session_state["filtro_valor"].strip().lower()
        campo = st.session_state["filtro_campo"]

        if filtro_txt:
            if campo == "Todos":
                mask = (
                    df_unique["Código"].astype(str).str.lower().str.contains(filtro_txt)
                    | df_unique["Descrição"].astype(str).str.lower().str.contains(filtro_txt)
                    | df_unique["Família"].astype(str).str.lower().str.contains(filtro_txt)
                )
            elif campo == "Código":
                mask = df_unique["Código"].astype(str).str.lower().str.contains(filtro_txt)
            elif campo == "Descrição":
                mask = df_unique["Descrição"].astype(str).str.lower().str.contains(filtro_txt)
            else:
                mask = df_unique["Família"].astype(str).str.lower().str.contains(filtro_txt)
            df_unique = df_unique[mask]

        df_unique = df_unique.sort_values(by="Código").reset_index(drop=True)

        updated_rows = []
        ajustes = st.session_state["ajustes_pecas"]
        resumo_ref = st.session_state["resumo_maquina_ref"]
        n_linhas_ref = int(resumo_ref.get("linhas_maquina", 1) or 1)

        # ---- Callbacks para sincronização bidirecional ----
        def cb_from_hect(kh, kr, tipo_lower, nlin):
            try:
                v = float(st.session_state[kh])
            except Exception:
                v = 0.0
            st.session_state[kr] = (v * nlin) if (tipo_lower == "linha") else v

        def cb_from_ref(kh, kr, tipo_lower, nlin):
            try:
                v = float(st.session_state[kr])
            except Exception:
                v = 0.0
            st.session_state[kh] = (v / nlin) if (tipo_lower == "linha" and nlin > 0) else v

        for _, row in df_unique.iterrows():
            codigo_item = format_codigo(row["Código"])
            st.markdown("---")
            st.subheader(f"{codigo_item} - {row['Descrição']}")

            # Valores atuais da base recalculada
            base_hect = float(row["hectare_proporcao_efetivo"])
            base_prop = int(row["proporcao_troca_%"])

            # Ajuste manual salvo (se houver)
            aj = ajustes.get(codigo_item, {})
            default_hect = float(aj["hect"]) if aj.get("manual_hect", False) and ("hect" in aj) else base_hect
            default_prop = int(aj["prop"]) if aj.get("manual_prop", False) and ("prop" in aj) else base_prop

            # Colunas e imagem
            has_img = "Imagem/url" in st.session_state["df_pecas_raw"].columns if st.session_state["df_pecas_raw"] is not None else False
            img_url = None
            if has_img:
                try:
                    raw = st.session_state["df_pecas_raw"]
                    img_match = raw[raw["Código"].apply(format_codigo) == codigo_item]
                    if not img_match.empty and isinstance(img_match.iloc[0].get("Imagem/url", None), str):
                        val = img_match.iloc[0]["Imagem/url"].strip()
                        if val and (val.startswith("http://") or val.startswith("https://")):
                            img_url = val
                except Exception:
                    img_url = None

            if img_url:
                cA, cB, cC, cD = st.columns([1.6, 1.2, 1.2, 1.2])
            else:
                cA, cB, cC = st.columns([2, 1, 1])
                cD = None

            with cA:
                st.write(f"Família: {row['Família']}")
                st.write(f"Custo unitário: {format_currency(row['custo_unitario'])}")
                st.write(f"Custo total: {format_currency(row['custo_total_base'])}")

            with cB:
                key_hect = f"hectare_prop_{codigo_item}"
                key_ref  = f"hect_ref_{codigo_item}"
                key_prop = f"prop_troca_{codigo_item}"

                # Inicializa estados (somente primeira vez)
                if key_hect not in st.session_state:
                    st.session_state[key_hect] = float(default_hect)

                tipo_prop_lower = str(row["Proporção"]).strip().lower()
                default_ref_calc = (float(st.session_state[key_hect]) * n_linhas_ref) if (tipo_prop_lower == "linha") else float(st.session_state[key_hect])

                if key_ref not in st.session_state:
                    st.session_state[key_ref] = float(default_ref_calc)

                if key_prop not in st.session_state:
                    st.session_state[key_prop] = int(default_prop)

                # Widgets com callbacks (Enter e +/- sincronizam)
                new_hectare_prop = st.number_input(
                    "Hectare/Proporção",
                    min_value=0.0,
                    step=1.0,
                    value=float(st.session_state[key_hect]),
                    key=key_hect,
                    on_change=lambda kh=key_hect, kr=key_ref, t=tipo_prop_lower, nl=n_linhas_ref: cb_from_hect(kh, kr, t, nl)
                )

                new_ref_input = st.number_input(
                    "Hectare referência",
                    min_value=0.0,
                    step=1.0,
                    value=float(st.session_state[key_ref]),
                    key=key_ref,
                    on_change=lambda kh=key_hect, kr=key_ref, t=tipo_prop_lower, nl=n_linhas_ref: cb_from_ref(kh, kr, t, nl)
                )

                new_prop_troca = st.slider(
                    "Proporção de troca (%)",
                    min_value=0, max_value=100,
                    value=int(st.session_state[key_prop]),
                    key=key_prop
                )

                # Usa os valores atuais do estado (já sincronizados pelas callbacks)
                synced_hect = float(st.session_state[key_hect])
                synced_ref  = float(st.session_state[key_ref])

                # Quantidade prevista
                vida_total, qtd_prevista = calcular_hect_ref_e_qtd_prevista(
                    row, resumo_ref, synced_hect, int(st.session_state[key_prop])
                )
                st.write(f"**Quantidade prevista**: {int(round(qtd_prevista))}")

            with cC:
                st.write(f"Proporção declarada: {row['Proporção']}")
                st.write(f"Qtd/Proporção: {row['Qtd/Proporção']}")
                st.write(f"Hectare/Proporção (original): {format_hectare_original(row['Hectare/Proporção'])}")
                st.write(f"Linhas do chassi (ref): {n_linhas_ref}")

            if cD is not None and img_url:
                with cD:
                    st.image(img_url, caption="Imagem da peça", use_container_width=True)

            # Persistência por código (manual flags)
            tol = 1e-9
            manual_hect = abs(float(synced_hect) - float(base_hect)) > tol
            manual_prop = int(st.session_state[key_prop]) != int(base_prop)

            if manual_hect or manual_prop or (codigo_item in ajustes):
                ajustes[codigo_item] = {
                    "hect": float(synced_hect) if manual_hect else ajustes.get(codigo_item, {}).get("hect"),
                    "prop": int(st.session_state[key_prop]) if manual_prop else ajustes.get(codigo_item, {}).get("prop"),
                    "manual_hect": manual_hect or ajustes.get(codigo_item, {}).get("manual_hect", False),
                    "manual_prop": manual_prop or ajustes.get(codigo_item, {}).get("manual_prop", False),
                }

            updated_rows.append({
                "Código": codigo_item,
                "hectare_proporcao_efetivo": float(synced_hect),
                "proporcao_troca_%": int(st.session_state[key_prop])
            })

        st.session_state["ajustes_pecas"] = ajustes

        # Aplica visualmente na base atual (reflete imediatamente)
        st.session_state["df_pecas_proc"]["Código"] = st.session_state["df_pecas_proc"]["Código"].apply(format_codigo)
        for u in updated_rows:
            m = st.session_state["df_pecas_proc"]["Código"] == u["Código"]
            st.session_state["df_pecas_proc"].loc[m, "hectare_proporcao_efetivo"] = u["hectare_proporcao_efetivo"]
            st.session_state["df_pecas_proc"].loc[m, "proporcao_troca_%"] = u["proporcao_troca_%"]

        st.session_state["df_pecas_proc"] = recalcular_pecas_pos_ajuste(
            st.session_state["df_pecas_proc"],
            resumo_ref
        )

        st.success("Ajustes aplicados. Vá para '3. Resumo / Resultados'.")


# ------------------------------------------------------------
# PÁGINA 3 - Resumo / Resultados
# ------------------------------------------------------------
elif pagina == "3. Resumo / Resultados":
    st.title("3. Resumo / Resultados")

    # Reprocessa aqui também (garante consistência no resumo)
    run_processamento_if_needed(show_msg=False)

    if (
        st.session_state["df_pecas_proc"] is None
        or st.session_state["df_maquinas_proc"] is None
        or st.session_state["df_pecas_proc"].empty
        or st.session_state["resumo_maquina_ref"] is None
    ):
        st.warning("Você ainda não carregou dados ou não fez os ajustes. Volte para as etapas anteriores.")
    else:
        resumo_ref = st.session_state["resumo_maquina_ref"]

        st.write(
            f"Modelo: {resumo_ref.get('modelo','-')} • Chassi ref: {resumo_ref.get('chassi_ref','-')} "
            f"• Linhas: {resumo_ref.get('linhas_maquina','?')} • Frota (modelo): {resumo_ref.get('n_chassis_frota', 1)}"
        )

        st.session_state["escopo_resumo"] = st.radio(
            "Escopo dos valores exibidos abaixo:",
            ["Apenas chassi selecionado", "Frota inteira"],
            horizontal=True,
            index=(0 if st.session_state["escopo_resumo"] == "Apenas chassi selecionado" else 1)
        )

        indicadores = calcular_indicadores_resumo(
            st.session_state["df_pecas_proc"],
            resumo_ref,
            escopo=st.session_state["escopo_resumo"]
        )

        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.metric(
                f"Custo total sugerido ({st.session_state['escopo_resumo']})",
                value=format_currency(indicadores['custo_total_estoque'])
            )
            cap = "Frota inteira do modelo." if st.session_state["escopo_resumo"] == "Frota inteira" else "Somente o chassi de referência."
            st.caption(cap)
        with col_r2:
            val_hect = (
                format_currency(indicadores['custo_medio_por_hectare'])
                if not np.isnan(indicadores['custo_medio_por_hectare'])
                else "n/d"
            )
            st.metric("Custo médio por hectare (R$/ha)", value=val_hect)
            st.caption("Base: máquina de referência (por máquina).")
        with col_r3:
            val_hora = (
                format_currency(indicadores['custo_medio_por_hora'])
                if not np.isnan(indicadores['custo_medio_por_hora'])
                else "n/d"
            )
            st.metric("Custo médio por hora (R$/h)", value=val_hora)
            st.caption("Base: máquina de referência (por máquina).")

        st.markdown("---")

        df_full_now = st.session_state["df_pecas_proc"].copy()
        familias_distintas_resumo = sorted(df_full_now["Família"].dropna().unique().tolist())
        familias_dropdown_resumo = ["Todos"] + familias_distintas_resumo

        st.session_state["filtro_familia_resumo"] = st.selectbox(
            "Família",
            familias_dropdown_resumo,
            index=(
                familias_dropdown_resumo.index(st.session_state["filtro_familia_resumo"])
                if st.session_state["filtro_familia_resumo"] in familias_dropdown_resumo
                else 0
            )
        )

        st.subheader(f"Consumo projetado de peças ({st.session_state['escopo_resumo'].lower()})")

        # Base numérica agregada
        df_export_preview_num = agregar_para_exportacao(
            st.session_state["df_pecas_proc"],
            resumo_ref,
            familia_filter=st.session_state["filtro_familia_resumo"],
            escopo=st.session_state["escopo_resumo"]
        ).copy()

        # -------- NOVO FILTRO TEXTO NA PÁGINA 3 --------
        if not df_export_preview_num.empty:
            cols_busca = df_export_preview_num.columns.tolist()
            opcoes_campo = ["Todos"] + cols_busca

            st.session_state["filtro_campo_resumo"] = st.selectbox(
                "Filtrar por campo",
                opcoes_campo,
                index=(
                    opcoes_campo.index(st.session_state["filtro_campo_resumo"])
                    if st.session_state["filtro_campo_resumo"] in opcoes_campo
                    else 0
                )
            )

            st.session_state["filtro_valor_resumo"] = st.text_input(
                "Texto para buscar (contém)",
                value=st.session_state["filtro_valor_resumo"]
            )

            filtro_txt_resumo = st.session_state["filtro_valor_resumo"].strip().lower()
            campo_resumo = st.session_state["filtro_campo_resumo"]

            if filtro_txt_resumo:
                if campo_resumo == "Todos":
                    mask = False
                    for c in cols_busca:
                        mask = mask | df_export_preview_num[c].astype(str).str.lower().str.contains(filtro_txt_resumo)
                else:
                    mask = df_export_preview_num[campo_resumo].astype(str).str.lower().str.contains(filtro_txt_resumo)
                df_export_preview_num = df_export_preview_num[mask]

        # Depois do filtro, arredonda Qtd recomendada para exibição
        if not df_export_preview_num.empty:
            df_export_preview_num["Qtd recomendada"] = (
                df_export_preview_num["Qtd recomendada"].apply(lambda x: int(round(x if pd.notna(x) else 0)))
            )

        st.dataframe(
            df_export_preview_num,
            column_config={
                "Código": st.column_config.TextColumn("Código"),
                "Descrição": st.column_config.TextColumn("Descrição"),
                "Família": st.column_config.TextColumn("Família"),
                "Qtd recomendada": st.column_config.NumberColumn("Qtd recomendada", format="%.0f"),
                "Custo total": st.column_config.NumberColumn("Custo total (R$)", format="R$ %.2f"),
            },
            hide_index=True,
            use_container_width=True,
        )

        with st.expander("Auditoria dos cálculos por item (debug)"):
            cods = st.session_state["df_pecas_proc"]["Código"].apply(format_codigo).tolist()
            if cods:
                cod_sel = st.selectbox("Selecione um Código para auditar", cods, index=0)
                row_item = st.session_state["df_pecas_proc"][
                    st.session_state["df_pecas_proc"]["Código"].apply(format_codigo) == cod_sel
                ].iloc[0]
                audit = auditar_item(row_item, resumo_ref)
                if audit:
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.write("**n_linhas**:", audit["n_linhas"])
                        st.write("**ha_ano_maquina**:", audit["ha_ano_maquina"])
                        st.write("**vida_por_linha_ou_maq**:", audit["vida_por_linha_ou_maq"])
                    with c2:
                        st.write("**vida_total**:", audit["vida_total"])
                        st.write("**qtd_por_linha_ou_maq**:", audit["qtd_por_linha_ou_maq"])
                        st.write("**qtd_total_por_ciclo**:", audit["qtd_total_por_ciclo"])
                    with c3:
                        st.write("**ciclos_ano**:", audit["ciclos_ano"])
                        st.write("**proporcao_troca_%**:", audit["proporcao_troca_%"])
                        st.write("**qtd_final**:", audit["qtd_final"])

        buffer_xlsx = gerar_planilha_exportacao(
            st.session_state["df_pecas_proc"],
            resumo_ref,
            familia_filter=st.session_state["filtro_familia_resumo"],
            escopo=st.session_state["escopo_resumo"]
        )

        st.download_button(
            label="Exportar Excel",
            data=buffer_xlsx,
            file_name="planejamento_manutencao.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
