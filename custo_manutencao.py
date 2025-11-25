import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import datetime
import altair as alt

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

        # mantido só por compatibilidade, mas não usado na lógica nova
        "prod_base": "Por máquina",

        "df_maquinas_proc": None,
        "resumo_maquina_ref": None,
        "df_pecas_proc": None,

        "filtro_campo": "Todos",
        "filtro_valor": "",
        "filtro_familia": "Todos",

        "filtro_familia_resumo": "Todos",
        "escopo_resumo": "Apenas chassi selecionado",

        # novos filtros página 3
        "filtro_campo_resumo": "Todos",
        "filtro_valor_resumo": "",

        # Persistência dos ajustes por Código
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

        # Modo global de cálculo da quantidade recomendada
        "modo_calculo_qtd": "Proporcional",

        # NOVO: considerar anos anteriores ou apenas ano atual
        "considerar_anos": "Considerar ano atual",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def aplicar_modo_operacao(valor_hectare_prop, modo):
    mults = st.session_state.get("multiplicadores_operacao",
                                 {"Leve": 1.5, "Moderado": 1.0, "Extremo": 0.6})
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
    modelo_escolhido, chassi_ref_escolhido
):
    """
    Sempre proporcional por largura:
    - hectare_ano_ref e largura_ref_m são a máquina de referência
    - para cada chassi: escala por largura_total_m (Linhas * Espaçamento)
    """
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
            "largura_maquina_m": 0.0,
            "anos_uso_maquina": 1,
            "considerar_anos": st.session_state.get("considerar_anos", "Considerar ano atual"),
        }
        return pd.DataFrame(), resumo_ref

    # Largura total por chassi
    df["largura_total_m"] = (df["Linhas"] * df["Espaçamento"]) / 100.0

    if largura_ref_m and largura_ref_m != 0:
        ha_hora_por_metro_ref = float(hectare_hora_ref or 0.0) / largura_ref_m
        ha_ano_por_metro_ref = float(hectare_ano_ref or 0.0) / largura_ref_m
    else:
        ha_hora_por_metro_ref = np.nan
        ha_ano_por_metro_ref = np.nan

    df["ha_hora_chassi"] = df["largura_total_m"] * ha_hora_por_metro_ref
    df["ha_ano_chassi"] = df["largura_total_m"] * ha_ano_por_metro_ref
    df["horas_chassi_ano"] = df["ha_ano_chassi"] / df["ha_hora_chassi"]

    # ------------------ Anos de uso ------------------
    considerar_anos_flag = st.session_state.get("considerar_anos", "Considerar ano atual")
    current_year = datetime.date.today().year

    if "Ano" not in df.columns:
        df["Ano"] = np.nan

    df["Ano"] = pd.to_numeric(df["Ano"], errors="coerce")

    if considerar_anos_flag == "Considerar anos anteriores":
        # anos_uso = (ano_atual - Ano) + 1, apenas se Ano <= ano_atual
        df["anos_uso"] = np.where(
            df["Ano"].notna() & (df["Ano"] <= current_year),
            (current_year - df["Ano"]).astype(int) + 1,
            np.nan
        )
        # Desconsidera máquinas sem ano ou com ano futuro
        df = df[df["anos_uso"].notna() & (df["anos_uso"] >= 1)]
        if df.empty:
            resumo_ref = {
                "chassi_ref": chassi_ref_escolhido if chassi_ref_escolhido else None,
                "linhas_maquina": 0,
                "ha_ano_maquina": 0.0,
                "ha_hora_maquina": 0.0,
                "horas_maquina_ano": 0.0,
                "n_chassis_frota": 0,
                "modelo": modelo_escolhido,
                "largura_maquina_m": 0.0,
                "anos_uso_maquina": 1,
                "considerar_anos": considerar_anos_flag,
            }
            return pd.DataFrame(), resumo_ref
    else:
        # Ignora ano: trata todas como ano de uso = 1
        df["anos_uso"] = 1

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
    largura_ref_maquina = float(linha_ref.get("largura_total_m", 0.0) or 0.0)
    anos_uso_ref = int(linha_ref.get("anos_uso", 1) or 1)

    resumo_ref = {
        "chassi_ref": chassi_ref_escolhido,
        "linhas_maquina": linhas_ref,
        "ha_ano_maquina": float(linha_ref["ha_ano_chassi"]),
        "ha_hora_maquina": float(linha_ref["ha_hora_chassi"]),
        "horas_maquina_ano": float(linha_ref["horas_chassi_ano"]) if pd.notna(linha_ref["horas_chassi_ano"]) else 0.0,
        "n_chassis_frota": int(len(df_sorted)),
        "modelo": modelo_escolhido,
        "largura_maquina_m": largura_ref_maquina,
        "anos_uso_maquina": anos_uso_ref,
        "considerar_anos": considerar_anos_flag,
    }

    return df_sorted, resumo_ref


# -------- helper: modo de cálculo por peça ou global --------

def _modo_qtd_para_codigo(row_or_codigo):
    """
    Retorna o modo de cálculo da quantidade:
      - se existir manual_modo=True e modo_qtd válido no ajustes_pecas => usa esse
      - senão => usa o modo global da página 1
    """
    ajustes = st.session_state.get("ajustes_pecas", {})
    try:
        if isinstance(row_or_codigo, str):
            cod = format_codigo(row_or_codigo)
        else:
            cod = format_codigo(row_or_codigo.get("Código"))
    except Exception:
        cod = None

    modo_global = st.session_state.get("modo_calculo_qtd", "Proporcional")

    if not cod or cod not in ajustes:
        return modo_global

    vals = ajustes[cod]
    if vals.get("manual_modo", False) and vals.get("modo_qtd") in ["Proporcional", "Inteiro"]:
        return vals["modo_qtd"]

    return modo_global


# -------- helper: cálculo por máquina específica --------

def _quantidade_para_maquina_especifica(row, ha_ano_maquina, n_linhas_maquina, anos_uso=1):
    """
    Calcula a quantidade recomendada de uma peça para UMA máquina específica,
    usando os parâmetros dessa máquina (hectares/ano e nº de linhas) e,
    opcionalmente, considerando anos anteriores.

    - Para "Considerar ano atual": usa apenas ha_ano_maquina (lógica original).
    - Para "Considerar anos anteriores" + modo Inteiro:
        conta somente os ciclos que "rompem" dentro do ano atual.
    - Para "Considerar anos anteriores" + modo Proporcional:
        conta (ciclos completos dentro do ano) + fração do ciclo em andamento
        ao final do ano atual (ex.: 3 trocas + fração proporcional ao restante).
    """
    try:
        vida_base = float(row["hectare_proporcao_efetivo"])  # durabilidade ajustada
        qtd_por_prop = float(row["Qtd/Proporção"])
        prop_troca = float(row["proporcao_troca_%"])
        tipo_prop = str(row["Proporção"]).strip().lower()
    except Exception:
        return 0.0

    if vida_base <= 0:
        return 0.0

    ha_ano = float(ha_ano_maquina or 0.0)
    n_linhas = int(n_linhas_maquina or 1)

    if ha_ano <= 0:
        return 0.0

    if tipo_prop == "linha":
        vida_total = vida_base * n_linhas
        qtd_total_por_ciclo = qtd_por_prop * n_linhas
    else:
        vida_total = vida_base
        qtd_total_por_ciclo = qtd_por_prop

    if vida_total <= 0:
        return 0.0

    modo_qtd = _modo_qtd_para_codigo(row)
    considerar_anos_flag = st.session_state.get("considerar_anos", "Considerar ano atual")

    if considerar_anos_flag == "Considerar anos anteriores" and anos_uso >= 1:
        # Hectares acumulados até o início e até o fim do ano atual
        start_prev = (anos_uso - 1) * ha_ano
        end_current = anos_uso * ha_ano

        if modo_qtd == "Inteiro":
            # Apenas ciclos completos que "rompem" dentro do ano atual
            ciclos_ini = np.floor(start_prev / vida_total)
            ciclos_fim = np.floor(end_current / vida_total)
            ciclos = max(0.0, ciclos_fim - ciclos_ini)
        else:
            # Proporcional: ciclos completos no ano + fração do ciclo em andamento
            x0 = start_prev / vida_total
            x1 = end_current / vida_total
            ciclos = max(0.0, x1 - np.floor(x0))
    else:
        # Lógica "por ano" (para ano atual ou quando não considera anos anteriores)
        ciclos_raw = ha_ano / vida_total
        if modo_qtd == "Inteiro":
            ciclos = np.floor(ciclos_raw)
        else:
            ciclos = ciclos_raw

    consumo_teorico = ciclos * qtd_total_por_ciclo
    qtd_rec = consumo_teorico * (prop_troca / 100.0)
    return float(qtd_rec)


def _quantidade_recomendada_uma_maquina(row, resumo_maquina_ref):
    """
    Usa a máquina de referência (resumo_maquina_ref) para calcular
    a quantidade recomendada (usada na página 2).
    """
    ha_ano = float(resumo_maquina_ref.get("ha_ano_maquina", 0.0) or 0.0)
    n_linhas = int(resumo_maquina_ref.get("linhas_maquina", 1) or 1)
    anos_uso_ref = int(resumo_maquina_ref.get("anos_uso_maquina", 1) or 1)
    return _quantidade_para_maquina_especifica(row, ha_ano, n_linhas, anos_uso_ref)


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

    # ---------------- Ajuste de Hectare/Proporção para Proporção = Máquina ----------------
    largura_ref = st.session_state.get("largura_ref_m") or 0.0
    largura_maquina_ref = float(resumo_maquina_ref.get("largura_maquina_m", 0.0) or 0.0)

    if largura_ref and largura_maquina_ref:
        fator_largura = largura_maquina_ref / largura_ref
    else:
        fator_largura = 1.0

    def _ajustar_hect_por_largura(row):
        try:
            val = float(row["Hectare/Proporção"])
        except Exception:
            return 0.0
        prop_str = str(row.get("Proporção", "")).strip().lower()
        # Para Proporção = Máquina, escala pela relação de larguras
        if prop_str in ["máquina", "maquina"]:
            val = val * fator_largura
        # Para Proporção = Linhas, mantém como está (vida por linha será tratada depois)
        return val

    # Base inicial (afetada por modo e multiplicadores + largura para Proporção=Máquina)
    df["hectare_proporcao_efetivo"] = df.apply(
        lambda r: aplicar_modo_operacao(_ajustar_hect_por_largura(r), st.session_state["modo_operacao"]),
        axis=1
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

    dedup_cols = ["Código", "Descrição", "Família", "Proporção", "Qtd/Proporção",
                  "hectare_proporcao_efetivo", "proporcao_troca_%", "custo_unitario"]
    dedup_cols = [c for c in dedup_cols if c in df.columns]
    df = df.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)

    return df


def recalcular_pecas_pos_ajuste(df_pecas_proc, resumo_maquina_ref):
    if df_pecas_proc is None or df_pecas_proc.empty:
        return df_pecas_proc
    df = df_pecas_proc.copy()

    # Reaplica somente os ajustes manuais (garantia extra)
    df = _reaplicar_ajustes(df)

    dedup_cols = ["Código", "Descrição", "Família", "Proporção", "Qtd/Proporção",
                  "hectare_proporcao_efetivo", "proporcao_troca_%", "custo_unitario"]
    dedup_cols = [c for c in dedup_cols if c in df.columns]
    df = df.drop_duplicates(subset=dedup_cols, keep="first")

    df["qtd_recomendada"] = df.apply(
        lambda r: _quantidade_recomendada_uma_maquina(r, resumo_maquina_ref),
        axis=1
    )
    df["custo_planejado_item"] = df["qtd_recomendada"] * df["custo_unitario"]
    return df


def agregar_para_exportacao(df_pecas_proc, resumo_maquina_ref, familia_filter="Todos", escopo="Apenas chassi selecionado"):
    """
    Agora calcula quantidade 'somando por chassi' de verdade, respeitando modo de cálculo
    e, quando configurado, anos de uso das máquinas.
    """
    if df_pecas_proc is None or df_pecas_proc.empty:
        return pd.DataFrame(columns=["Código", "Descrição", "Família", "Qtd recomendada", "Custo total"])

    df_escalada = df_pecas_proc.copy()
    if familia_filter != "Todos":
        df_escalada = df_escalada[df_escalada["Família"] == familia_filter]

    cols_dedup = ["Código", "Descrição", "Família", "Proporção", "Qtd/Proporção",
                  "hectare_proporcao_efetivo", "proporcao_troca_%", "custo_unitario"]
    cols_dedup = [c for c in cols_dedup if c in df_escalada.columns]
    df_escalada = df_escalada.drop_duplicates(subset=cols_dedup, keep="first").reset_index(drop=True)

    df_maqs = st.session_state.get("df_maquinas_proc")
    if df_maqs is None or df_maqs.empty:
        # fallback: comportamento antigo (multiplica por n_chassis)
        n_chassis_total = int(resumo_maquina_ref.get("n_chassis_frota", 1) or 1)
        n_chassis = 1 if escopo == "Apenas chassi selecionado" else n_chassis_total

        df_escalada["Qtd recomendada"] = df_escalada["qtd_recomendada"] * n_chassis
        df_escalada["Custo total"] = df_escalada["Qtd recomendada"] * df_escalada["custo_unitario"]
    else:
        df_maqs_local = df_maqs.copy()
        df_maqs_local["Chassi"] = df_maqs_local["Chassi"].astype(str)

        chassi_sel = st.session_state.get("chassi_selecionado")

        if escopo == "Apenas chassi selecionado":
            if chassi_sel and chassi_sel != "Todos":
                df_maqs_local = df_maqs_local[df_maqs_local["Chassi"] == str(chassi_sel)]
            else:
                df_maqs_local = df_maqs_local.sort_values(by="Chassi").head(1)
        else:
            # Frota inteira: usa todos os chassis do modelo
            pass

        qts = []
        custos = []
        for _, p in df_escalada.iterrows():
            qtd_total_frota = 0.0
            for _, m in df_maqs_local.iterrows():
                ha_ano_maq = float(m.get("ha_ano_chassi", 0.0) or 0.0)
                n_linhas_maq = int(m.get("Linhas", 1) or 1)
                anos_uso_maq = int(m.get("anos_uso", 1) or 1)
                qtd_maq = _quantidade_para_maquina_especifica(p, ha_ano_maq, n_linhas_maq, anos_uso_maq)
                qtd_total_frota += qtd_maq

            qts.append(qtd_total_frota)
            custos.append(qtd_total_frota * float(p["custo_unitario"] or 0.0))

        df_escalada["Qtd recomendada"] = qts
        df_escalada["Custo total"] = custos

    agr = (
        df_escalada
        .groupby("Código", as_index=False, sort=False)
        .agg({
            "Descrição": "first",
            "Família": "first",
            "Qtd recomendada": "sum",
            "Custo total": "sum"
        })
        .sort_values(by="Código")
        .reset_index(drop=True)
    )
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

    df_agr = agregar_para_exportacao(
        df_pecas_proc,
        resumo_maquina_ref,
        familia_filter="Todos",
        escopo=escopo
    )
    custo_total_escopo = df_agr["Custo total"].sum() if not df_agr.empty else 0.0

    ha_ano = resumo_maquina_ref.get("ha_ano_maquina", 0.0)
    horas_ano = resumo_maquina_ref.get("horas_maquina_ano", 0.0)

    custo_por_maquina_ref = df_pecas_proc["custo_planejado_item"].sum()

    custo_medio_por_hectare = (custo_por_maquina_ref / ha_ano) if ha_ano else np.nan
    custo_medio_por_hora = (custo_por_maquina_ref / horas_ano) if horas_ano else np.nan

    return {
        "custo_total_estoque": float(custo_total_escopo),
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

    ciclos_raw = ha_ano / vida_total if vida_total > 0 else 0
    modo_qtd = _modo_qtd_para_codigo(row_item)
    if modo_qtd == "Inteiro":
        ciclos = np.floor(ciclos_raw)
    else:
        ciclos = ciclos_raw

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

def calcular_hect_ref_e_qtd_prevista(
    row,
    resumo_maquina_ref,
    hectare_efetivo_atual,
    proporcao_troca_atual,
    modo_qtd_atual=None
):
    """
    Calcula:
      - Hectare referência (vida_total): se Proporção='Linha' => hectare_efetivo * n_linhas; senão, por máquina.
      - Quantidade prevista: usa ha_ano do chassi selecionado, vida_total e Qtd/Proporção,
        aplicando a proporção de troca informada no input atual.

    Se modo_qtd_atual for informado (página 2), usa esse modo.
    Se não, busca modo em ajustes_pecas/global via _modo_qtd_para_codigo.

    Quando "Considerar anos anteriores" está ativo, utiliza também anos_uso_maquina
    para separar o que aconteceu antes e dentro do ano atual, tanto no modo Inteiro
    quanto no Proporcional (ex.: 3 trocas + parte proporcional ao restante).
    """
    try:
        tipo_prop = str(row["Proporção"]).strip().lower()
        n_linhas = int(resumo_maquina_ref.get("linhas_maquina", 1) or 1)
        qtd_por_prop = float(row["Qtd/Proporção"])
        ha_ano = float(resumo_maquina_ref.get("ha_ano_maquina", 0.0) or 0.0)
        vida_base = float(hectare_efetivo_atual)
        prop_troca = float(proporcao_troca_atual)
        anos_uso_ref = int(resumo_maquina_ref.get("anos_uso_maquina", 1) or 1)
    except Exception:
        return 0.0, 0.0

    considerar_anos_flag = st.session_state.get("considerar_anos", "Considerar ano atual")

    if tipo_prop == "linha":
        vida_total = vida_base * n_linhas
        qtd_total_por_ciclo = qtd_por_prop * n_linhas
    else:
        vida_total = vida_base
        qtd_total_por_ciclo = qtd_por_prop

    if vida_total > 0:
        if modo_qtd_atual is not None:
            modo_qtd = modo_qtd_atual
        else:
            modo_qtd = _modo_qtd_para_codigo(row)

        if considerar_anos_flag == "Considerar anos anteriores" and anos_uso_ref >= 1:
            # Mesma lógica usada em _quantidade_para_maquina_especifica
            start_prev = (anos_uso_ref - 1) * ha_ano
            end_current = anos_uso_ref * ha_ano

            if modo_qtd == "Inteiro":
                ciclos_ini = np.floor(start_prev / vida_total)
                ciclos_fim = np.floor(end_current / vida_total)
                ciclos = max(0.0, ciclos_fim - ciclos_ini)
            else:
                x0 = start_prev / vida_total
                x1 = end_current / vida_total
                ciclos = max(0.0, x1 - np.floor(x0))
        else:
            ciclos_raw = ha_ano / vida_total
            if modo_qtd == "Inteiro":
                ciclos = np.floor(ciclos_raw)
            else:
                ciclos = ciclos_raw

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
      Código, Hectare/Proporção, Proporção de troca (%), Modo de cálculo
    usando os valores ATUAIS da página 2 (já com ajustes/modo).
    """
    df = st.session_state.get("df_pecas_proc")
    if df is None or df.empty:
        return pd.DataFrame(columns=["Código", "Hectare/Proporção", "Proporção de troca (%)", "Modo de cálculo"])
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

    # acrescenta modo de cálculo (Proporcional/Inteiro) por código
    base["Modo de cálculo"] = base["Código"].apply(lambda c: _modo_qtd_para_codigo(c))
    return base


def gerar_planilha_ajustes():
    """
    Gera buffer Excel com os ajustes atuais (todos os códigos, inclusive sem ajustes manuais),
    incluindo a coluna 'Modo de cálculo'.
    """
    df_exp = montar_df_ajustes_atual()
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df_exp.to_excel(writer, index=False, sheet_name="Ajustes")
    buffer.seek(0)
    return buffer


def aplicar_importacao_ajustes(df_import):
    """
    Aplica em massa os ajustes vindos do Excel:
      - Código
      - Hectare/Proporção
      - Proporção de troca (%)
      - (Opcional) Modo de cálculo: Proporcional / Inteiro

    Marca hectare/prop como MANUAL e atualiza modo_qtd quando informado.
    Recalcula base da página 2.
    """
    if df_import is None or df_import.empty:
        return False, "Arquivo vazio ou inválido."

    # Mapeia colunas por case-insensitive
    cols = {c.strip().lower(): c for c in df_import.columns}
    # obrigatórias
    req = {
        "código": None,
        "hectare/proporção": None,
        "proporção de troca (%)": None
    }
    # opcional
    opt_modo_col = None

    for k in list(req.keys()):
        if k in cols:
            req[k] = cols[k]
    if None in req.values():
        return False, "As colunas obrigatórias são: Código, Hectare/Proporção, Proporção de troca (%)."

    if "modo de cálculo" in cols:
        opt_modo_col = cols["modo de cálculo"]

    df = df_import[[req["código"], req["hectare/proporção"], req["proporção de troca (%)"]]].copy()
    df.columns = ["Código", "Hectare/Proporção", "Proporção de troca (%)"]

    if opt_modo_col is not None:
        df["Modo de cálculo"] = df_import[opt_modo_col]
        df["Modo de cálculo"] = (
            df["Modo de cálculo"]
            .astype(str)
            .str.strip()
            .str.capitalize()
            .replace({"Proporcional": "Proporcional", "Inteiro": "Inteiro"})
        )
        df.loc[~df["Modo de cálculo"].isin(["Proporcional", "Inteiro"]), "Modo de cálculo"] = np.nan
    else:
        df["Modo de cálculo"] = np.nan

    df["Código"] = df["Código"].apply(format_codigo)
    df["Hectare/Proporção"] = pd.to_numeric(df["Hectare/Proporção"], errors="coerce").fillna(0.0)
    df["Proporção de troca (%)"] = pd.to_numeric(df["Proporção de troca (%)"], errors="coerce").fillna(0.0).astype(int)

    ajustes = st.session_state.get("ajustes_pecas", {}).copy()
    modo_global = st.session_state.get("modo_calculo_qtd", "Proporcional")

    for _, r in df.iterrows():
        cod = r["Código"]
        antigo = ajustes.get(cod, {})
        modo_importado = r["Modo de cálculo"] if isinstance(r.get("Modo de cálculo", np.nan), str) and r["Modo de cálculo"] in ["Proporcional", "Inteiro"] else antigo.get("modo_qtd", None)

        if modo_importado is None:
            modo_importado = antigo.get("modo_qtd", modo_global)

        manual_modo = (modo_importado != modo_global)

        ajustes[cod] = {
            "hect": float(r["Hectare/Proporção"]),
            "prop": int(r["Proporção de troca (%)"]),
            "manual_hect": True,
            "manual_prop": True,
            "modo_qtd": modo_importado,
            "manual_modo": manual_modo,
        }

    st.session_state["ajustes_pecas"] = ajustes

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
                                 {"Leve": 1.5, "Moderado": 1.0, "Extremo": 0.6})
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
        st.session_state.get("modo_calculo_qtd"),
        st.session_state.get("considerar_anos"),
        float(mults.get("Leve", 1.5)),
        float(mults.get("Moderado", 1.0)),
        float(mults.get("Extremo", 0.6)),
        int(st.session_state.get("default_proporcao_troca", 50)),
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
    Inclui modo_calculo_qtd e considerar_anos, então mudar Proporcional/Inteiro
    na página 1 ou alternar entre 'ano atual' e 'anos anteriores'
    força reprocessamento.
    Também reseta os rádios de modo das peças que NÃO são manuais.
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

        # sincroniza os rádios de modo com o global para quem NÃO é manual
        ajustes = st.session_state.get("ajustes_pecas", {})
        for cod, vals in ajustes.items():
            if not vals.get("manual_modo", False):
                key = f"modo_qtd_{format_codigo(cod)}"
                if key in st.session_state:
                    del st.session_state[key]

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
    ["1. Entrada de Dados", "2. Ajustes de Peças", "3. Resumo / Resultados", "4. Análise operacional"]
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
            "Hectare médio por ano (máquina de referência)",
            min_value=0,
            step=1,
            value=st.session_state["hectare_ano_ref"] if st.session_state["hectare_ano_ref"] else 0
        )

        st.session_state["hectare_hora_ref"] = st.number_input(
            "Hectares por hora (máquina de referência)",
            min_value=0.0,
            step=0.1,
            value=st.session_state["hectare_hora_ref"] if st.session_state["hectare_hora_ref"] else 0.0
        )

    with col_in2:
        st.session_state["largura_ref_m"] = st.number_input(
            "Largura do equipamento de referência (m)",
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
                                            value=float(mults.get("Leve", 1.5)))
        with cB:
            mults["Moderado"] = st.number_input("Multiplicador - Moderado",
                                                min_value=0.1, max_value=5.0, step=0.1,
                                                value=float(mults.get("Moderado", 1.0)))
        with cC:
            mults["Extremo"] = st.number_input("Multiplicador - Extremo",
                                               min_value=0.1, max_value=5.0, step=0.1,
                                               value=float(mults.get("Extremo", 0.6)))
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

    # NOVO: horizonte de cálculo (anos anteriores x ano atual)
    st.markdown("---")
    st.subheader("Horizonte de cálculo (vida útil das máquinas)")
    st.session_state["considerar_anos"] = st.radio(
        "Como considerar o ano das máquinas?",
        ["Considerar ano atual", "Considerar anos anteriores"],
        horizontal=True,
        index=(
            0 if st.session_state["considerar_anos"] == "Considerar ano atual" else 1
        )
    )

    # NOVO: modo de cálculo da quantidade
    st.markdown("---")
    st.subheader("Modo de cálculo da quantidade de peças")

    st.session_state["modo_calculo_qtd"] = st.radio(
        "Como calcular a quantidade recomendada de peças?",
        ["Proporcional", "Inteiro"],
        horizontal=True,
        index=(0 if st.session_state["modo_calculo_qtd"] == "Proporcional" else 1)
    )

    st.caption(
        "- **Proporcional**: usa frações de ciclo (ex.: 1,5x da quantidade recomendada).\n"
        "- **Inteiro**: só considera ciclos completos (ex.: 1x até completar 2x a vida da peça)."
    )

    st.markdown("---")

    run_processamento_if_needed(show_msg=True)


# ------------------------------------------------------------
# PÁGINA 2 - Ajustes de Peças
# ------------------------------------------------------------
elif pagina == "2. Ajustes de Peças":
    st.title("2. Ajustes de Peças")

    run_processamento_if_needed(show_msg=False)

    if (
        st.session_state["df_pecas_proc"] is None
        or st.session_state["df_maquinas_proc"] is None
        or st.session_state["df_pecas_proc"].empty
        or st.session_state["resumo_maquina_ref"] is None
    ):
        st.warning("Primeiro importe os dados e processe na página '1. Entrada de Dados'.")
    else:
        resumo_ref = st.session_state["resumo_maquina_ref"]
        # NOVO: mostrar chassi de referência na página 2
        st.write(
            f"Modelo: {resumo_ref.get('modelo','-')} • Chassi ref: {resumo_ref.get('chassi_ref','-')} "
            f"• Linhas: {resumo_ref.get('linhas_maquina','?')} • Frota (modelo): {resumo_ref.get('n_chassis_frota', 1)}"
        )

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
                    "Importar ajustes (.xlsx) com colunas: Código, Hectare/Proporção, Proporção de troca (%), Modo de cálculo (opcional)",
                    type=["xlsx"],
                    key="upload_ajustes_xlsx"
                )

                if up_file is not None:
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
        n_linhas_ref = int(resumo_ref.get("linhas_maquina", 1) or 1)

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

        modo_global = st.session_state.get("modo_calculo_qtd", "Proporcional")

        for _, row in df_unique.iterrows():
            codigo_item = format_codigo(row["Código"])
            st.markdown("---")
            st.subheader(f"{codigo_item} - {row['Descrição']}")

            base_hect = float(row["hectare_proporcao_efetivo"])
            base_prop = int(row["proporcao_troca_%"])

            aj = ajustes.get(codigo_item, {})
            manual_modo_flag = aj.get("manual_modo", False)

            if manual_modo_flag and aj.get("modo_qtd") in ["Proporcional", "Inteiro"]:
                default_modo = aj["modo_qtd"]
            else:
                default_modo = modo_global

            default_hect = float(aj["hect"]) if aj.get("manual_hect", False) and ("hect" in aj) else base_hect
            default_prop = int(aj["prop"]) if aj.get("manual_prop", False) and ("prop" in aj) else base_prop

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

                key_modo = f"modo_qtd_{codigo_item}"
                # Inicializa só se ainda não existe; assim o usuário pode sobrescrever
                if key_modo not in st.session_state:
                    st.session_state[key_modo] = default_modo

                modo_escolhido = st.radio(
                    "Modo de cálculo",
                    ["Proporcional", "Inteiro"],
                    horizontal=True,
                    key=key_modo
                )

            with cB:
                key_hect = f"hectare_prop_{codigo_item}"
                key_ref = f"hect_ref_{codigo_item}"
                key_prop = f"prop_troca_{codigo_item}"

                if key_hect not in st.session_state:
                    st.session_state[key_hect] = float(default_hect)

                tipo_prop_lower = str(row["Proporção"]).strip().lower()
                default_ref_calc = (float(st.session_state[key_hect]) * n_linhas_ref) if (tipo_prop_lower == "linha") else float(st.session_state[key_hect])

                if key_ref not in st.session_state:
                    st.session_state[key_ref] = float(default_ref_calc)

                if key_prop not in st.session_state:
                    st.session_state[key_prop] = int(default_prop)

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

                synced_hect = float(st.session_state[key_hect])
                synced_ref = float(st.session_state[key_ref])

                # Calcula vida_total (Hectare referência) e quantidade prevista
                vida_total, qtd_prevista = calcular_hect_ref_e_qtd_prevista(
                    row,
                    resumo_ref,
                    synced_hect,
                    int(st.session_state[key_prop]),
                    modo_qtd_atual=modo_escolhido
                )
                st.write(f"**Quantidade prevista**: {int(round(qtd_prevista))}")

            with cC:
                st.write(f"Proporção declarada: {row['Proporção']}")
                st.write(f"Qtd/Proporção: {row['Qtd/Proporção']}")
                st.write(f"Hectare/Proporção (original): {format_hectare_original(row['Hectare/Proporção'])}")
                st.write(f"Linhas do chassi (ref): {n_linhas_ref}")

                # Horas = Hectare referência (vida_total) / Hectares por hora da máquina de referência
                ha_hora_maquina = float(resumo_ref.get("ha_hora_maquina", 0.0) or 0.0)
                if ha_hora_maquina > 0:
                    horas_peca = vida_total / ha_hora_maquina
                    st.write(f"Horas: {horas_peca:.2f}")
                else:
                    st.write("Horas: n/d")

            if cD is not None and img_url:
                with cD:
                    st.image(img_url, caption="Imagem da peça", use_container_width=True)

            tol = 1e-9
            manual_hect = abs(float(synced_hect) - float(base_hect)) > tol
            manual_prop = int(st.session_state[key_prop]) != int(base_prop)
            manual_modo = (modo_escolhido != modo_global)

            antigo = ajustes.get(codigo_item, {})

            ajustes[codigo_item] = {
                "hect": float(synced_hect) if manual_hect else antigo.get("hect"),
                "prop": int(st.session_state[key_prop]) if manual_prop else antigo.get("prop"),
                "manual_hect": manual_hect or antigo.get("manual_hect", False),
                "manual_prop": manual_prop or antigo.get("manual_prop", False),
                "modo_qtd": modo_escolhido,
                "manual_modo": manual_modo,
            }

            updated_rows.append({
                "Código": codigo_item,
                "hectare_proporcao_efetivo": float(synced_hect),
                "proporcao_troca_%": int(st.session_state[key_prop])
            })

        st.session_state["ajustes_pecas"] = ajustes

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

        df_export_preview_num = agregar_para_exportacao(
            st.session_state["df_pecas_proc"],
            resumo_ref,
            familia_filter=st.session_state["filtro_familia_resumo"],
            escopo=st.session_state["escopo_resumo"]
        ).copy()

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


# ------------------------------------------------------------
# PÁGINA 4 - Análise operacional
# ------------------------------------------------------------
elif pagina == "4. Análise operacional":
    st.title("4. Análise operacional")

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

        st.markdown("---")

        # ---------------- Parâmetros operacionais da página 4 ----------------
        col_p1, col_p2 = st.columns(2)

        with col_p1:
            # Horas de trabalho por dia – número com 2 casas decimais
            if "horas_trabalho_dia" not in st.session_state:
                st.session_state["horas_trabalho_dia"] = 0.0

            horas_trabalho_dia = st.number_input(
                "Horas de trabalho por dia",
                min_value=0.0,
                step=0.25,
                format="%.2f",
                value=float(st.session_state["horas_trabalho_dia"])
            )
            st.session_state["horas_trabalho_dia"] = float(horas_trabalho_dia)

        with col_p2:
            # Início da operação – date_input com calendário
            if "inicio_operacao" not in st.session_state:
                st.session_state["inicio_operacao"] = datetime.date.today()
            inicio_operacao = st.date_input(
                "Início da operação",
                value=st.session_state["inicio_operacao"],
                key="inicio_operacao_date"
            )
            st.session_state["inicio_operacao"] = inicio_operacao

        # ---------------- Cálculos derivados (referência) ----------------
        hectare_ano_ref = float(resumo_ref.get("ha_ano_maquina", 0.0) or 0.0)
        hectare_hora_ref = float(resumo_ref.get("ha_hora_maquina", 0.0) or 0.0)

        if hectare_ano_ref > 0 and hectare_hora_ref > 0:
            horas_total_operacao = hectare_ano_ref / hectare_hora_ref
        else:
            horas_total_operacao = 0.0

        if horas_total_operacao > 0 and horas_trabalho_dia > 0:
            total_dias = horas_total_operacao / horas_trabalho_dia
        else:
            total_dias = 0.0

        # Fim da operação = início + total_dias (arredondado)
        if isinstance(inicio_operacao, datetime.date):
            fim_operacao = inicio_operacao + datetime.timedelta(days=int(round(total_dias)))
        else:
            fim_operacao = None

        # Hectare por dia da máquina de referência (base)
        hectare_por_dia_ref = hectare_hora_ref * horas_trabalho_dia

        if hectare_ano_ref <= 0 or hectare_hora_ref <= 0:
            st.warning("Verifique os parâmetros da página 1 (Hectare médio por ano, Hectares por hora e largura).")

        st.markdown("---")

        # ---------------- Escopo: Chassi específico x Frota inteira ----------------
        if "escopo_operacional" not in st.session_state:
            st.session_state["escopo_operacional"] = "Chassi específico"

        escopo_label = st.radio(
            "Escopo para cálculo das quantidades:",
            ["Chassi específico", "Frota inteira"],
            horizontal=True,
            index=(0 if st.session_state["escopo_operacional"] == "Chassi específico" else 1)
        )
        st.session_state["escopo_operacional"] = escopo_label

        df_pecas = st.session_state["df_pecas_proc"].copy()
        df_maqs_all = st.session_state["df_maquinas_proc"].copy()
        df_maqs_all["Chassi"] = df_maqs_all["Chassi"].astype(str)
        chassi_sel = st.session_state.get("chassi_selecionado")

        if escopo_label == "Chassi específico":
            if chassi_sel and chassi_sel != "Todos":
                df_maqs_local = df_maqs_all[df_maqs_all["Chassi"] == str(chassi_sel)]
            else:
                df_maqs_local = df_maqs_all.sort_values(by="Chassi").head(1)
        else:
            # Frota inteira -> todos os chassis do modelo já filtrados em df_maquinas_proc
            df_maqs_local = df_maqs_all.copy()

        if df_maqs_local.empty:
            st.warning("Não há chassis disponíveis para o escopo selecionado.")
        else:
            # ---------------- Resumo operacional (com Hectare total e escopo) ----------------
            # Calcula Hectare por dia conforme o escopo
            if escopo_label == "Chassi específico":
                # Não soma: usa apenas a máquina de referência
                hectare_por_dia = hectare_por_dia_ref
            else:
                # Frota inteira: soma Hectare por dia de cada máquina do escopo
                hectare_por_dia = 0.0
                for _, m in df_maqs_local.iterrows():
                    ha_hora_maq = float(m.get("ha_hora_chassi", 0.0) or 0.0)
                    hectare_por_dia += ha_hora_maq * horas_trabalho_dia

            # Hectare total = Total de dias * Hectare por dia (respeitando escopo)
            hectare_total = total_dias * hectare_por_dia

            st.markdown("### Resumo operacional (máquina de referência)")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric(
                    "Horas total de operação",
                    f"{horas_total_operacao:.2f}"
                )
            with c2:
                st.metric(
                    "Total de dias",
                    f"{total_dias:.2f}"
                )
            with c3:
                st.metric(
                    "Hectare por dia",
                    f"{hectare_por_dia:.2f}"
                )
            with c4:
                st.metric(
                    "Hectare total",
                    f"{hectare_total:.2f}"
                )

            col_d1, col_d2 = st.columns(2)
            inicio_str = (
                inicio_operacao.strftime("%d/%m/%Y")
                if isinstance(inicio_operacao, datetime.date) else "-"
            )
            fim_str = (
                fim_operacao.strftime("%d/%m/%Y")
                if isinstance(fim_operacao, datetime.date) else "-"
            )
            with col_d1:
                st.write(f"**Início da operação:** {inicio_str}")
            with col_d2:
                st.write(f"**Fim da operação (ref.):** {fim_str}")

            if horas_trabalho_dia <= 0:
                st.warning("Informe um valor positivo em 'Horas de trabalho por dia' para gerar o calendário.")
            else:
                linhas_calendario = []

                considerar_anos_flag = st.session_state.get("considerar_anos", "Considerar ano atual")

                # ---------- Geração de eventos por máquina + peça ----------
                for _, m in df_maqs_local.iterrows():
                    ha_ano_maq = float(m.get("ha_ano_chassi", 0.0) or 0.0)
                    ha_hora_maq = float(m.get("ha_hora_chassi", 0.0) or 0.0)
                    n_linhas_maq = int(m.get("Linhas", 1) or 1)
                    anos_uso_maq = int(m.get("anos_uso", 1) or 1)

                    if ha_ano_maq <= 0 or ha_hora_maq <= 0:
                        continue

                    hectare_por_dia_maq = ha_hora_maq * horas_trabalho_dia
                    if hectare_por_dia_maq <= 0:
                        continue

                    # Hectares acumulados antes do ano atual e dentro do ano atual
                    if considerar_anos_flag == "Considerar anos anteriores" and anos_uso_maq > 1:
                        start_prev_ha = (anos_uso_maq - 1) * ha_ano_maq
                    else:
                        start_prev_ha = 0.0
                    end_current_ha = start_prev_ha + ha_ano_maq

                    for _, row_p in df_pecas.iterrows():
                        try:
                            cod = format_codigo(row_p["Código"])
                            fam = row_p["Família"]
                            desc = row_p["Descrição"]
                            vida_base = float(row_p["hectare_proporcao_efetivo"])
                            qtd_por_prop = float(row_p["Qtd/Proporção"])
                            prop_troca = float(row_p["proporcao_troca_%"])
                            tipo_prop = str(row_p["Proporção"]).strip().lower()
                            custo_unit = float(row_p.get("custo_unitario", 0.0) or 0.0)
                        except Exception:
                            continue

                        if vida_base <= 0 or qtd_por_prop <= 0 or prop_troca <= 0:
                            continue

                        # vida_total e quantidade por ciclo (teórica)
                        if tipo_prop == "linha":
                            vida_total_ha = vida_base * n_linhas_maq
                            qtd_ciclo_teorico = qtd_por_prop * n_linhas_maq
                        else:
                            vida_total_ha = vida_base
                            qtd_ciclo_teorico = qtd_por_prop

                        if vida_total_ha <= 0 or qtd_ciclo_teorico <= 0:
                            continue

                        # Quantidade recomendada para ESSA máquina (ano atual), usando lógica de anos_uso
                        qtd_recomendada_maq = _quantidade_para_maquina_especifica(
                            row_p, ha_ano_maq, n_linhas_maq, anos_uso_maq
                        )
                        if qtd_recomendada_maq <= 0:
                            continue

                        # Quantidade "cheia" recomendada em cada ciclo (já considerando proporção de troca)
                        q_evento_cheio = qtd_ciclo_teorico * (prop_troca / 100.0)
                        if q_evento_cheio <= 0:
                            # fallback simples: tudo em um único evento na data inicial
                            quantidades = [qtd_recomendada_maq]
                            offsets_ha = [0.0]
                        else:
                            # Eventos completos (ciclos que "rompem" dentro do ano atual)
                            k_start = int(np.floor(start_prev_ha / vida_total_ha)) + 1
                            k_end = int(np.floor(end_current_ha / vida_total_ha))
                            if k_end >= k_start:
                                full_ks = list(range(k_start, k_end + 1))
                            else:
                                full_ks = []

                            quantidades = []
                            offsets_ha = []

                            for k in full_ks:
                                A_k = k * vida_total_ha
                                offset_ha = A_k - start_prev_ha
                                offsets_ha.append(offset_ha)
                                quantidades.append(q_evento_cheio)

                            total_full = sum(quantidades) if quantidades else 0.0
                            resto = float(qtd_recomendada_maq - total_full)

                            # Se houver resto (modo Proporcional ou arredondamentos),
                            # lança em um último evento no final do ano atual
                            if resto > 1e-6:
                                quantidades.append(resto)
                                offsets_ha.append(end_current_ha - start_prev_ha)

                            if not quantidades:
                                # fallback: tudo em um único evento no final do ano
                                quantidades = [qtd_recomendada_maq]
                                offsets_ha = [end_current_ha - start_prev_ha]

                        # Gera eventos para esta máquina e esta peça
                        for q_evt, off_ha in zip(quantidades, offsets_ha):
                            if isinstance(inicio_operacao, datetime.date):
                                if hectare_por_dia_maq > 0:
                                    dias_offset = off_ha / hectare_por_dia_maq
                                else:
                                    dias_offset = 0.0
                                data_evt = inicio_operacao + datetime.timedelta(days=int(round(dias_offset)))
                                data_troca_str = data_evt.strftime("%m/%Y")
                            else:
                                data_evt = None
                                data_troca_str = ""

                            custo_evt = q_evt * custo_unit

                            linhas_calendario.append({
                                "Família": fam,
                                "Código": cod,
                                "Descrição": desc,
                                "Data troca": data_troca_str,
                                "Quantidade peça": q_evt,
                                "Custo": custo_evt
                            })

                # ---------- AGREGAÇÃO POR Data troca / Código / Família ----------
                df_cal = pd.DataFrame(linhas_calendario)

                if not df_cal.empty:
                    group_cols = ["Família", "Código", "Descrição", "Data troca"]
                    df_cal = (
                        df_cal
                        .groupby(group_cols, as_index=False)
                        .agg({
                            "Quantidade peça": "sum",
                            "Custo": "sum"
                        })
                    )

                    # Converte "Data troca" (mm/aaaa) para datetime (01/mm/aaaa) para ordenar corretamente
                    df_cal["Data troca"] = pd.to_datetime(
                        "01/" + df_cal["Data troca"].astype(str),
                        format="%d/%m/%Y",
                        errors="coerce"
                    )
                    # Remove linhas inválidas e ordena por data
                    df_cal = df_cal.dropna(subset=["Data troca"])
                    df_cal = df_cal.sort_values("Data troca").reset_index(drop=True)

                    # Arredonda quantidade para visualização
                    df_cal["Quantidade peça"] = df_cal["Quantidade peça"].round(0)

                    # ---------------- Filtros (Família + campo livre, similar páginas 2 e 3) ----------------
                    familias_p4 = sorted(df_cal["Família"].dropna().unique().tolist())
                    familias_dropdown_p4 = ["Todos"] + familias_p4

                    if "filtro_familia_p4" not in st.session_state:
                        st.session_state["filtro_familia_p4"] = "Todos"
                    if "filtro_campo_p4" not in st.session_state:
                        st.session_state["filtro_campo_p4"] = "Todos"
                    if "filtro_valor_p4" not in st.session_state:
                        st.session_state["filtro_valor_p4"] = ""

                    col_f1, col_f2, col_f3 = st.columns([1.2, 1.2, 2])
                    with col_f1:
                        st.session_state["filtro_familia_p4"] = st.selectbox(
                            "Família",
                            familias_dropdown_p4,
                            index=(
                                familias_dropdown_p4.index(st.session_state["filtro_familia_p4"])
                                if st.session_state["filtro_familia_p4"] in familias_dropdown_p4
                                else 0
                            )
                        )
                    with col_f2:
                        st.session_state["filtro_campo_p4"] = st.selectbox(
                            "Filtrar por campo",
                            ["Todos", "Código", "Descrição", "Data troca"],
                            index=(
                                ["Todos", "Código", "Descrição", "Data troca"].index(st.session_state["filtro_campo_p4"])
                                if st.session_state["filtro_campo_p4"] in ["Todos", "Código", "Descrição", "Data troca"]
                                else 0
                            )
                        )
                    with col_f3:
                        st.session_state["filtro_valor_p4"] = st.text_input(
                            "Valor do filtro (contém)",
                            value=st.session_state["filtro_valor_p4"]
                        )

                    # Aplica filtro de família
                    fam_sel_p4 = st.session_state["filtro_familia_p4"]
                    if fam_sel_p4 != "Todos":
                        df_cal = df_cal[df_cal["Família"] == fam_sel_p4]

                    # Aplica filtro de texto
                    filtro_txt_p4 = st.session_state["filtro_valor_p4"].strip().lower()
                    campo_p4 = st.session_state["filtro_campo_p4"]

                    if filtro_txt_p4:
                        if campo_p4 == "Todos":
                            mask = (
                                df_cal["Código"].astype(str).str.lower().str.contains(filtro_txt_p4)
                                | df_cal["Descrição"].astype(str).str.lower().str.contains(filtro_txt_p4)
                                | df_cal["Data troca"].dt.strftime("%m/%Y").str.lower().str.contains(filtro_txt_p4)
                            )
                        elif campo_p4 == "Código":
                            mask = df_cal["Código"].astype(str).str.lower().str.contains(filtro_txt_p4)
                        elif campo_p4 == "Descrição":
                            mask = df_cal["Descrição"].astype(str).str.lower().str.contains(filtro_txt_p4)
                        else:  # Data troca
                            mask = df_cal["Data troca"].dt.strftime("%m/%Y").str.lower().str.contains(filtro_txt_p4)
                        df_cal = df_cal[mask]

                    st.markdown("### Calendário de trocas projetado")

                    st.dataframe(
                        df_cal,
                        column_config={
                            "Família": st.column_config.TextColumn("Família"),
                            "Código": st.column_config.TextColumn("Código"),
                            "Descrição": st.column_config.TextColumn("Descrição"),
                            "Data troca": st.column_config.DateColumn("Data troca (mm/aaaa)", format="MM/YYYY"),
                            "Quantidade peça": st.column_config.NumberColumn("Quantidade peça", format="%.0f"),
                            "Custo": st.column_config.NumberColumn("Custo (R$)", format="R$ %.2f"),
                        },
                        hide_index=True,
                        use_container_width=True,
                    )

                    # ---------------- Exportar tabela da página 4 ----------------
                    if not df_cal.empty:
                        # Cria uma cópia para export, mantendo as colunas iguais
                        df_export_p4 = df_cal.copy()
                        buffer_p4 = BytesIO()
                        with pd.ExcelWriter(buffer_p4, engine="xlsxwriter") as writer:
                            df_export_p4.to_excel(writer, index=False, sheet_name="Analise_operacional")
                        buffer_p4.seek(0)

                        st.download_button(
                            label="Exportar tabela da análise operacional (Excel)",
                            data=buffer_p4,
                            file_name="analise_operacional.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )

                    # ---------------- Gráfico de colunas ----------------
                    if not df_cal.empty:
                        st.markdown("### Gráfico de trocas por mês/ano")

                        metrica_graf = st.radio(
                            "Métrica do gráfico",
                            ["Quantidade de peças", "Custo (R$)"],
                            horizontal=True
                        )

                        if metrica_graf == "Quantidade de peças":
                            y_col = "Quantidade peça"
                            y_title = "Quantidade de peças"
                        else:
                            y_col = "Custo"
                            y_title = "Custo (R$)"

                        chart_data = df_cal.copy()
                        # Colunas formatadas para tooltip
                        chart_data["Quantidade_str"] = chart_data["Quantidade peça"].apply(
                            lambda x: f"{int(round(x))}"
                        )
                        chart_data["Custo_str"] = chart_data["Custo"].apply(format_currency)
                        # rótulo categórico mm/aaaa para usar apenas datas que têm informação
                        chart_data["DataLabel"] = chart_data["Data troca"].dt.strftime("%m/%Y")

                        chart = (
                            alt.Chart(chart_data)
                            .mark_bar()
                            .encode(
                                x=alt.X(
                                    "DataLabel:N",
                                    title="Mês/Ano",
                                    sort=alt.SortField(field="Data troca", order="ascending")
                                ),
                                y=alt.Y(f"{y_col}:Q", title=y_title),
                                tooltip=[
                                    alt.Tooltip("DataLabel:N", title="Mês/Ano"),
                                    alt.Tooltip("Quantidade_str:N", title="Quantidade"),
                                    alt.Tooltip("Custo_str:N", title="Custo"),
                                ]
                            )
                            .interactive()
                        )

                        st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("Nenhum evento de troca foi gerado com os parâmetros atuais.")
