import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import datetime
import altair as alt
import math

st.set_page_config(
    page_title="Calculate Parts",
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

        # Parâmetros globais ajustáveis (fallback quando a peça não traz Proporção de troca na planilha)
        "default_proporcao_troca": 50,

        # ✅ AJUSTE SOLICITADO: Leve=1.30 / Extremo=0.70
        "multiplicadores_operacao": {"Leve": 1.30, "Moderado": 1.00, "Extremo": 0.70},

        # Estado para importação de ajustes
        "ajustes_import_df": None,
        "ajustes_import_filename": None,
        "ajustes_import_applied": False,

        # Horizonte de cálculo (vida útil das máquinas)
        "considerar_anos": "Considerar todas as máquinas como novas",

        # ---------------------------
        # Página 5 - Plano de Manutenção
        # ---------------------------
        "plano_chassi_selecionado": None,   # chassi escolhido só para o plano (independente da pág 1)
        "plano_tempo_operacao_anos": 1,     # inteiro

        "filtro_familia_p5": "Todos",
        "filtro_campo_p5": "Todos",
        "filtro_valor_p5": "",

        "metrica_graf_p5": "Custo total (R$)",  # ou "Custo por hectare (R$/ha)"

        # ---------------------------
        # Página 6 - Confiabilidade
        # ---------------------------
        "df_beta_raw": None,
        "beta_import_filename": None,
        "codigo_confiabilidade_sel": None,

    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _get_mult_modo(modo: str) -> float:
    mults = st.session_state.get(
        "multiplicadores_operacao",
        {"Leve": 1.30, "Moderado": 1.00, "Extremo": 0.70}
    )
    try:
        return float(mults.get(modo, 1.0))
    except Exception:
        return 1.0


def aplicar_modo_operacao(valor_hectare_prop, modo):
    mult = _get_mult_modo(modo)
    try:
        return float(valor_hectare_prop) * mult
    except Exception:
        return 0.0


def format_currency(v):
    if pd.isna(v):
        return "R$ 0,00"
    try:
        return "R$ " + f"{float(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
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
    except Exception:
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

    # ✅ NOVO: higieniza "Proporção de troca (%)" (0..100)
    if "Proporção de troca (%)" in df.columns:
        df["Proporção de troca (%)"] = pd.to_numeric(df["Proporção de troca (%)"], errors="coerce")
        df["Proporção de troca (%)"] = df["Proporção de troca (%)"].clip(lower=0, upper=100)

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
    """
    Higieniza a tabela de máquinas, incluindo a coluna 'Estado' (Usado/Novo).
    Qualquer valor diferente ou ausente será tratado como 'Novo'.
    """
    dfm = df_maquinas.copy()
    for col in ["Modelo", "Chassi"]:
        if col in dfm.columns:
            dfm[col] = dfm[col].astype(str).str.strip()
    for col in ["Linhas", "Espaçamento", "Ano"]:
        if col in dfm.columns:
            dfm[col] = pd.to_numeric(dfm[col], errors="coerce")

    if "Estado" in dfm.columns:
        dfm["Estado"] = dfm["Estado"].astype(str).str.strip().str.capitalize()
        dfm.loc[~dfm["Estado"].isin(["Usado", "Novo"]), "Estado"] = "Novo"
    else:
        dfm["Estado"] = "Novo"

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

    Horizonte de cálculo:
    - "Considerar todas as máquinas como novas" => anos_uso = 1 para todas
    - "Considerar com base no ano e estado":
        * Estado = "Usado" => anos_uso = (ano_atual - Ano) + 1  (se Ano <= ano_atual)
        * Estado = "Novo" ou Ano inválido => anos_uso = 1
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
            "considerar_anos": st.session_state.get("considerar_anos", "Considerar todas as máquinas como novas"),
        }
        return pd.DataFrame(), resumo_ref

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

    considerar_anos_flag = st.session_state.get("considerar_anos", "Considerar todas as máquinas como novas")
    current_year = datetime.date.today().year

    if "Ano" not in df.columns:
        df["Ano"] = np.nan

    df["Ano"] = pd.to_numeric(df["Ano"], errors="coerce")

    if "Estado" not in df.columns:
        df["Estado"] = "Novo"
    else:
        df["Estado"] = df["Estado"].astype(str).str.strip().str.capitalize()
        df.loc[~df["Estado"].isin(["Usado", "Novo"]), "Estado"] = "Novo"

    if considerar_anos_flag == "Considerar com base no ano e estado":
        anos_uso_list = []
        for _, r in df.iterrows():
            estado = r.get("Estado", "Novo")
            ano_maq = r.get("Ano", np.nan)
            if estado == "Usado" and not pd.isna(ano_maq) and ano_maq <= current_year:
                anos_uso = int(current_year - int(ano_maq)) + 1
                anos_uso = max(1, anos_uso)
            else:
                anos_uso = 1
            anos_uso_list.append(anos_uso)
        df["anos_uso"] = anos_uso_list
    else:
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


# -------- helper: regra fixa do sistema --------
def _modo_qtd_para_codigo(row_or_codigo):
    # Regra fixa do sistema: sempre Inteiro (não expor no app)
    return "Inteiro"


# -------- helper: cálculo por máquina específica --------
def _quantidade_para_maquina_especifica(row, ha_ano_maquina, n_linhas_maquina, anos_uso=1):
    try:
        vida_base = float(row["hectare_proporcao_efetivo"])
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

    # ✅ Sempre Inteiro
    considerar_anos_flag = st.session_state.get("considerar_anos", "Considerar todas as máquinas como novas")

    if considerar_anos_flag == "Considerar com base no ano e estado" and anos_uso >= 1:
        start_prev = (anos_uso - 1) * ha_ano
        end_current = anos_uso * ha_ano
        ciclos_ini = np.floor(start_prev / vida_total)
        ciclos_fim = np.floor(end_current / vida_total)
        ciclos = max(0.0, ciclos_fim - ciclos_ini)
    else:
        ciclos_raw = ha_ano / vida_total
        ciclos = np.floor(ciclos_raw)

    consumo_teorico = ciclos * qtd_total_por_ciclo
    qtd_rec = consumo_teorico * (prop_troca / 100.0)
    return float(qtd_rec)


# -------- helper NOVO: página 5 (sempre máquina nova e simulação Ano 1..N) --------
def _quantidade_para_maquina_especifica_plano(row, ha_ano_maquina, n_linhas_maquina, ano_ciclo):
    try:
        vida_base = float(row["hectare_proporcao_efetivo"])
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

    # ✅ Sempre Inteiro (por ano/ciclo)
    anos_uso = int(max(1, ano_ciclo))
    start_prev = (anos_uso - 1) * ha_ano
    end_current = anos_uso * ha_ano
    ciclos_ini = np.floor(start_prev / vida_total)
    ciclos_fim = np.floor(end_current / vida_total)
    ciclos = max(0.0, ciclos_fim - ciclos_ini)

    consumo_teorico = ciclos * qtd_total_por_ciclo
    qtd_rec = consumo_teorico * (prop_troca / 100.0)
    return float(qtd_rec)


def _quantidade_recomendada_uma_maquina(row, resumo_maquina_ref):
    ha_ano = float(resumo_maquina_ref.get("ha_ano_maquina", 0.0) or 0.0)
    n_linhas = int(resumo_maquina_ref.get("linhas_maquina", 1) or 1)
    anos_uso_ref = int(resumo_maquina_ref.get("anos_uso_maquina", 1) or 1)
    return _quantidade_para_maquina_especifica(row, ha_ano, n_linhas, anos_uso_ref)


# ---------------- REAPLICAÇÃO DE AJUSTES (respeita apenas o que foi MANUAL) ----------------
def _reaplicar_ajustes(df):
    """
    Reaplica somente os campos ajustados MANUALMENTE em st.session_state['ajustes_pecas'].

    ✅ Importação (backup):
    - Armazena 'hect_base' (base antes do modo). Esse ajuste é aplicado
      na construção do DF (construir_df_pecas).
    - Aqui reaplicamos apenas o que é manual em nível de 'efetivo' (manual_hect=True) e a proporção.
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

        # manual_hect=True significa "travado" (não sofre modo)
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
        if prop_str in ["máquina", "maquina"]:
            val = val * fator_largura
        return val

    # ✅ Base antes do modo (após largura quando Proporção=Máquina)
    df["hectare_prop_base"] = df.apply(_ajustar_hect_por_largura, axis=1)

    # ✅ Aplicar override "base" vindo da importação (sem travar o modo)
    ajustes = st.session_state.get("ajustes_pecas", {})
    if isinstance(ajustes, dict) and not df.empty and "Código" in df.columns:
        df["Código"] = df["Código"].apply(format_codigo)

        for cod, vals in ajustes.items():
            if not isinstance(vals, dict):
                continue
            cod_norm = format_codigo(cod)
            m = df["Código"] == cod_norm
            if not m.any():
                continue

            # Se existir hect_base manual, ele substitui a base (antes do modo)
            if vals.get("manual_hect_base", False) and (vals.get("hect_base") is not None):
                try:
                    df.loc[m, "hectare_prop_base"] = float(vals["hect_base"])
                except Exception:
                    pass

    # ✅ Agora aplica o modo de operação sobre a base (funciona com/sem import)
    df["hectare_proporcao_efetivo"] = df["hectare_prop_base"].apply(
        lambda v: aplicar_modo_operacao(v, st.session_state["modo_operacao"])
    )

    # ✅ NOVO: proporção de troca padrão por peça (vinda da planilha Peças) com fallback no default global
    default_global = float(st.session_state.get("default_proporcao_troca", 50))
    if "Proporção de troca (%)" in df.columns:
        prop_col = pd.to_numeric(df["Proporção de troca (%)"], errors="coerce")
        prop_col = prop_col.clip(lower=0, upper=100)
        df["proporcao_troca_%"] = prop_col.fillna(default_global).astype(float)
    else:
        df["proporcao_troca_%"] = float(default_global)

    df["custo_unitario"] = df["Custo"]
    df["custo_total_base"] = df["Qtd/Proporção"] * df["custo_unitario"]

    # Reaplica somente ajustes manuais (efetivo travado e proporção)
    df = _reaplicar_ajustes(df)

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

def _format_tamanho_bi(linhas, espacamento):
    """
    Regras:
    - com linhas e espaçamento: '7.45'
    - com linhas e sem espaçamento: '7'
    """
    try:
        linhas_str = str(int(float(linhas))) if pd.notna(linhas) else ""
    except Exception:
        linhas_str = str(linhas).strip() if linhas is not None else ""

    try:
        if pd.isna(espacamento) or str(espacamento).strip() == "":
            return linhas_str

        esp_float = float(espacamento)

        # Se for inteiro exato, mostra sem casas
        if esp_float.is_integer():
            esp_str = str(int(esp_float))
        else:
            esp_str = str(esp_float).replace(".", ",")  # opcional para exibição BR
            esp_str = esp_str.replace(",", ".")         # mantém formato pedido: 7.45

        return f"{linhas_str}.{esp_str}"
    except Exception:
        return linhas_str


def _aplicar_modo_operacao_em_df_bi(df_base, modo):
    """
    Recalcula somente o hectare_proporcao_efetivo conforme o modo,
    preservando a base já ajustada e a proporção de troca da peça.
    """
    df = df_base.copy()

    # Usa hectare_prop_base quando existir; senão usa o efetivo atual como fallback
    if "hectare_prop_base" in df.columns:
        df["hectare_proporcao_efetivo"] = df["hectare_prop_base"].apply(
            lambda v: aplicar_modo_operacao(v, modo)
        )
    else:
        # fallback defensivo
        mult_atual = _get_mult_modo(st.session_state.get("modo_operacao", "Moderado"))
        mult_novo = _get_mult_modo(modo)
        if mult_atual == 0:
            mult_atual = 1.0

        df["hectare_proporcao_efetivo"] = (
            pd.to_numeric(df["hectare_proporcao_efetivo"], errors="coerce").fillna(0.0) / mult_atual
        ) * mult_novo

    return df


def gerar_base_bi_plano_manutencao(chassi_p5, tempo_anos):
    """
    Gera a base completa para BI com os 3 modos empilhados:
    Leve, Moderado e Extremo.
    """
    df_maqs_proc = st.session_state.get("df_maquinas_proc")
    df_maqs_raw = st.session_state.get("df_maquinas_raw")
    df_pecas_proc = st.session_state.get("df_pecas_proc")

    if (
        df_maqs_proc is None or df_maqs_proc.empty or
        df_pecas_proc is None or df_pecas_proc.empty
    ):
        return pd.DataFrame()

    df_maqs_proc = df_maqs_proc.copy()
    df_maqs_proc["Chassi"] = df_maqs_proc["Chassi"].astype(str)

    row_proc = df_maqs_proc[df_maqs_proc["Chassi"] == str(chassi_p5)]
    if row_proc.empty:
        return pd.DataFrame()
    row_proc = row_proc.iloc[0]

    row_raw = None
    if df_maqs_raw is not None and not df_maqs_raw.empty and "Chassi" in df_maqs_raw.columns:
        tmp_raw = df_maqs_raw.copy()
        tmp_raw["Chassi"] = tmp_raw["Chassi"].astype(str)
        rr = tmp_raw[tmp_raw["Chassi"] == str(chassi_p5)]
        if not rr.empty:
            row_raw = rr.iloc[0]

    modelo = row_raw.get("Modelo") if row_raw is not None else row_proc.get("Modelo")
    linhas = row_raw.get("Linhas") if row_raw is not None else row_proc.get("Linhas")
    espacamento = row_raw.get("Espaçamento") if row_raw is not None else row_proc.get("Espaçamento")
    tamanho = _format_tamanho_bi(linhas, espacamento)

    ha_ano_maq = float(row_proc.get("ha_ano_chassi", 0.0) or 0.0)
    ha_hora_maq = float(row_proc.get("ha_hora_chassi", 0.0) or 0.0)
    n_linhas_maq = int(row_proc.get("Linhas", 1) or 1)

    if ha_ano_maq <= 0:
        return pd.DataFrame()

    horas_ano_maq = (ha_ano_maq / ha_hora_maq) if ha_hora_maq > 0 else 0.0

    # Base única por código
    df_base = df_pecas_proc.copy()
    df_base = df_base.groupby("Código", as_index=False).first().reset_index(drop=True)
    df_base["Código"] = df_base["Código"].apply(format_codigo)

    modos = ["Leve", "Moderado", "Extremo"]
    linhas_out = []

    for modo in modos:
        df_modo = _aplicar_modo_operacao_em_df_bi(df_base, modo)

        for ano_ciclo in range(1, int(tempo_anos) + 1):
            ano_label = f"Ano {ano_ciclo}"
            hectare_acumulado = ha_ano_maq * ano_ciclo
            horas_acumuladas = horas_ano_maq * ano_ciclo

            for _, p in df_modo.iterrows():
                qtd = _quantidade_para_maquina_especifica_plano(
                    p, ha_ano_maq, n_linhas_maq, ano_ciclo
                )

                if qtd <= 0:
                    continue

                custo_unit = float(p.get("custo_unitario", 0.0) or 0.0)
                custo_total = float(qtd) * custo_unit

                linhas_out.append({
                    "Modelo": modelo,
                    "Tamanho": tamanho,
                    "Modo operação": modo,
                    "Ano": ano_label,
                    "Hectare": float(hectare_acumulado),
                    "Horas": float(horas_acumuladas),
                    "Família": p.get("Família", ""),
                    "Código": format_codigo(p.get("Código", "")),
                    "Descrição": p.get("Descrição", ""),
                    "Qtd recomendada": int(round(qtd)),
                    "Custo total (R$)": float(custo_total),
                })

    df_bi = pd.DataFrame(linhas_out)

    if df_bi.empty:
        return df_bi

    def _ano_num(x):
        try:
            return int(str(x).lower().replace("ano", "").strip())
        except Exception:
            return 999999

    ordem_modo = {"Leve": 1, "Moderado": 2, "Extremo": 3}
    df_bi["__ord_modo"] = df_bi["Modo operação"].map(ordem_modo).fillna(999)
    df_bi["__ord_ano"] = df_bi["Ano"].apply(_ano_num)

    df_bi = df_bi.sort_values(
        by=["__ord_modo", "__ord_ano", "Família", "Código"],
        ascending=[True, True, True, True]
    ).reset_index(drop=True)

    df_bi = df_bi.drop(columns=["__ord_modo", "__ord_ano"])

    return df_bi


def gerar_excel_bi_plano_manutencao(chassi_p5, tempo_anos):
    df_bi = gerar_base_bi_plano_manutencao(chassi_p5, tempo_anos)

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df_bi.to_excel(writer, index=False, sheet_name="Dados_BI")
    buffer.seek(0)
    return buffer

def agregar_para_exportacao(df_pecas_proc, resumo_maquina_ref, familia_filter="Todos", escopo="Apenas chassi selecionado"):
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
        return {"custo_total_estoque": 0.0, "custo_medio_por_hectare": 0.0, "custo_medio_por_hora": 0.0}

    df_agr = agregar_para_exportacao(df_pecas_proc, resumo_maquina_ref, familia_filter="Todos", escopo=escopo)
    custo_total_escopo = df_agr["Custo total"].sum() if not df_agr.empty else 0.0

    if escopo == "Frota inteira":
        df_maqs = st.session_state.get("df_maquinas_proc")
        if df_maqs is not None and not df_maqs.empty:
            total_ha_ano = pd.to_numeric(df_maqs.get("ha_ano_chassi", 0.0), errors="coerce").fillna(0.0).sum()
            total_horas_ano = pd.to_numeric(df_maqs.get("horas_chassi_ano", 0.0), errors="coerce").fillna(0.0).sum()
        else:
            total_ha_ano = 0.0
            total_horas_ano = 0.0

        custo_medio_por_hectare = (custo_total_escopo / total_ha_ano) if total_ha_ano else np.nan
        custo_medio_por_hora = (custo_total_escopo / total_horas_ano) if total_horas_ano else np.nan
    else:
        ha_ano = float(resumo_maquina_ref.get("ha_ano_maquina", 0.0) or 0.0)
        horas_ano = float(resumo_maquina_ref.get("horas_maquina_ano", 0.0) or 0.0)
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
    # ✅ sempre Inteiro
    ciclos = np.floor(ciclos_raw)

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
    proporcao_troca_atual
):
    """
    ✅ Regra fixa: sempre Inteiro (não exposto no sistema).
    Retorna: vida_total, qtd_prevista, ciclos
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
        return 0.0, 0.0, 0.0

    considerar_anos_flag = st.session_state.get("considerar_anos", "Considerar todas as máquinas como novas")

    if tipo_prop == "linha":
        vida_total = vida_base * n_linhas
        qtd_total_por_ciclo = qtd_por_prop * n_linhas
    else:
        vida_total = vida_base
        qtd_total_por_ciclo = qtd_por_prop

    qtd_prevista = 0.0
    ciclos = 0.0

    if vida_total > 0:
        if considerar_anos_flag == "Considerar com base no ano e estado" and anos_uso_ref >= 1:
            start_prev = (anos_uso_ref - 1) * ha_ano
            end_current = anos_uso_ref * ha_ano
            ciclos_ini = np.floor(start_prev / vida_total)
            ciclos_fim = np.floor(end_current / vida_total)
            ciclos = max(0.0, ciclos_fim - ciclos_ini)
        else:
            ciclos_raw = ha_ano / vida_total
            ciclos = np.floor(ciclos_raw)

        consumo_teorico = ciclos * qtd_total_por_ciclo
        qtd_prevista = consumo_teorico * (prop_troca / 100.0)

    return float(vida_total), float(qtd_prevista), float(ciclos)


# ------------------------------------------------------------
# === AJUSTES: EXPORT/IMPORT (sem Modo de cálculo) ===
# ------------------------------------------------------------
def montar_df_ajustes_atual():
    """
    Exporta os valores ATUAIS (efetivos na página 2):
      Código, Hectare/Proporção, Proporção de troca (%)
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
    df_exp = montar_df_ajustes_atual()
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df_exp.to_excel(writer, index=False, sheet_name="Ajustes")
    buffer.seek(0)
    return buffer


def aplicar_importacao_ajustes(df_import):
    """
    ✅ Regras:
    - Importação NÃO trava o modo de operação.
    - O 'Hectare/Proporção' importado (efetivo) é convertido em BASE,
      dividindo pelo multiplicador do modo atual, e armazenado como 'hect_base'.
    - Importação NÃO considera (nem aceita) "Modo de cálculo" (regra fixa: Inteiro).
    """
    if df_import is None or df_import.empty:
        return False, "Arquivo vazio ou inválido."

    cols = {c.strip().lower(): c for c in df_import.columns}
    req = {"código": None, "hectare/proporção": None, "proporção de troca (%)": None}

    for k in list(req.keys()):
        if k in cols:
            req[k] = cols[k]
    if None in req.values():
        return False, "As colunas obrigatórias são: Código, Hectare/Proporção, Proporção de troca (%)."

    df = df_import[[req["código"], req["hectare/proporção"], req["proporção de troca (%)"]]].copy()
    df.columns = ["Código", "Hectare/Proporção", "Proporção de troca (%)"]

    df["Código"] = df["Código"].apply(format_codigo)
    df["Hectare/Proporção"] = pd.to_numeric(df["Hectare/Proporção"], errors="coerce").fillna(0.0)
    df["Proporção de troca (%)"] = pd.to_numeric(df["Proporção de troca (%)"], errors="coerce").fillna(0.0).astype(int)

    ajustes = st.session_state.get("ajustes_pecas", {}).copy()

    # ✅ Converte efetivo importado para BASE (antes do modo)
    mult_atual = _get_mult_modo(st.session_state.get("modo_operacao", "Moderado"))
    if mult_atual == 0:
        mult_atual = 1.0

    for _, r in df.iterrows():
        cod = r["Código"]
        antigo = ajustes.get(cod, {})

        hect_efetivo_import = float(r["Hectare/Proporção"])
        hect_base_import = hect_efetivo_import / mult_atual

        ajustes[cod] = {
            # ✅ base antes do modo (não trava o modo)
            "hect_base": float(hect_base_import),
            "manual_hect_base": True,

            # mantém compatibilidade com ajuste manual efetivo (se existir)
            "hect": antigo.get("hect", None),
            "manual_hect": antigo.get("manual_hect", False),

            "prop": int(r["Proporção de troca (%)"]),
            "manual_prop": True,
        }

    st.session_state["ajustes_pecas"] = ajustes

    # Reprocessa para refletir imediatamente (modo continua funcional)
    run_processamento_if_needed(show_msg=False)

    return True, "Importação aplicada com sucesso (sem travar o Modo de operação)."


# ------------------------------------------------------------
# Confiabilidade Weibull - helpers
# ------------------------------------------------------------
def higienizar_beta(df_beta):
    df = df_beta.copy()
    cols_map = {str(c).strip().lower(): c for c in df.columns}

    obrigatorias = {
        "código": None,
        "família": None,
        "descrição": None,
        "beta": None,
    }

    for k in obrigatorias.keys():
        if k in cols_map:
            obrigatorias[k] = cols_map[k]

    if obrigatorias["código"] is None or obrigatorias["beta"] is None:
        raise ValueError("A planilha Beta precisa ter, no mínimo, as colunas 'Código' e 'Beta'.")

    rename_map = {
        obrigatorias["código"]: "Código",
        obrigatorias["beta"]: "Beta",
    }
    if obrigatorias["família"] is not None:
        rename_map[obrigatorias["família"]] = "Família"
    if obrigatorias["descrição"] is not None:
        rename_map[obrigatorias["descrição"]] = "Descrição"

    df = df.rename(columns=rename_map).copy()

    if "Família" not in df.columns:
        df["Família"] = ""
    if "Descrição" not in df.columns:
        df["Descrição"] = ""

    df = df[["Código", "Família", "Descrição", "Beta"]].copy()
    df["Código"] = df["Código"].apply(format_codigo)
    df["Beta"] = pd.to_numeric(df["Beta"], errors="coerce")
    df = df.dropna(subset=["Código", "Beta"])
    df = df[df["Beta"] > 0].copy()
    df = df.drop_duplicates(subset=["Código"], keep="last").reset_index(drop=True)
    return df


def obter_base_confiabilidade():
    """
    Usa a tabela de Peças importada na Página 1 como origem da vida média.
    A vida base é Hectare/Proporção, com ajuste de largura quando Proporção = Máquina.
    Depois, os modos Leve/Moderado/Extremo são aplicados sobre essa base.
    """
    df_pecas_raw = st.session_state.get("df_pecas_raw")
    modelo_sel = st.session_state.get("modelo_selecionado")

    if df_pecas_raw is None or df_pecas_raw.empty:
        return pd.DataFrame()

    df = higienizar_pecas(df_pecas_raw).copy()

    if modelo_sel and "Modelo" in df.columns:
        df = df[df["Modelo"] == modelo_sel].copy()

    if df.empty:
        return pd.DataFrame()

    largura_ref = st.session_state.get("largura_ref_m") or 0.0
    resumo_ref = st.session_state.get("resumo_maquina_ref") or {}
    largura_maquina_ref = float(resumo_ref.get("largura_maquina_m", 0.0) or 0.0)

    if largura_ref and largura_maquina_ref:
        fator_largura = largura_maquina_ref / largura_ref
    else:
        fator_largura = 1.0

    def _calc_base(row):
        try:
            val = float(row["Hectare/Proporção"])
        except Exception:
            return np.nan

        prop_str = str(row.get("Proporção", "")).strip().lower()
        if prop_str in ["máquina", "maquina"]:
            val = val * fator_largura
        return val

    df["mu_base"] = df.apply(_calc_base, axis=1)
    df["Código"] = df["Código"].apply(format_codigo)

    # aplica ajustes de base importados/manualizados, quando existirem
    ajustes = st.session_state.get("ajustes_pecas", {})
    if isinstance(ajustes, dict) and not df.empty:
        for cod, vals in ajustes.items():
            if not isinstance(vals, dict):
                continue
            cod_norm = format_codigo(cod)
            m = df["Código"] == cod_norm
            if not m.any():
                continue
            if vals.get("manual_hect_base", False) and (vals.get("hect_base") is not None):
                try:
                    df.loc[m, "mu_base"] = float(vals["hect_base"])
                except Exception:
                    pass

    cols_keep = [c for c in ["Modelo", "Código", "Família", "Descrição", "Proporção", "Hectare/Proporção", "mu_base"] if c in df.columns]
    df = df[cols_keep].copy()
    df = df.dropna(subset=["Código", "mu_base"])
    df = df[df["mu_base"] > 0].copy()
    df = df.drop_duplicates(subset=["Código"], keep="first").reset_index(drop=True)
    return df


def montar_base_confiabilidade(df_beta):
    """
    Retorna:
    1) df_relatorio -> base completa com descrição (para gráfico e conferência)
    2) df_export -> base enxuta para exportação/BI:
       Modelo, Modo operação, Código, R(t), Hectare, Eta, Beta
    """
    if df_beta is None or df_beta.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_base = obter_base_confiabilidade()
    if df_base.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_beta = higienizar_beta(df_beta)

    df_merge = df_base.merge(df_beta[["Código", "Beta"]], on="Código", how="inner")

    if df_merge.empty:
        return pd.DataFrame(), pd.DataFrame()

    modos = ["Leve", "Moderado", "Extremo"]
    linhas_out = []

    for _, row in df_merge.iterrows():
        try:
            codigo = format_codigo(row["Código"])
            modelo = row.get("Modelo", "")
            descricao = row.get("Descrição", "")
            beta = float(row["Beta"])
            mu_base = float(row["mu_base"])
        except Exception:
            continue

        if beta <= 0 or mu_base <= 0:
            continue

        # malha comum por código, baseada no Moderado
        mu_moderado = aplicar_modo_operacao(mu_base, "Moderado")
        hectare_max = 2.0 * mu_moderado
        passo = hectare_max / 10.0
        hectares = [round(i * passo, 10) for i in range(11)]

        for modo in modos:
            mu_modo = aplicar_modo_operacao(mu_base, modo)
            if mu_modo <= 0:
                continue

            try:
                eta = mu_modo / math.gamma(1 + 1 / beta)
            except Exception:
                continue

            for hect in hectares:
                try:
                    r_t = math.exp(-((float(hect) / float(eta)) ** beta)) if eta > 0 else np.nan
                except Exception:
                    r_t = np.nan

                linhas_out.append({
                    "Modelo": modelo,
                    "Modo operação": modo,
                    "Código": codigo,
                    "Descrição": descricao,
                    "Hectare": float(hect),
                    "Beta": float(beta),
                    "Eta": float(eta),
                    "R(t)": float(r_t) if pd.notna(r_t) else np.nan,
                })

    df_rel = pd.DataFrame(linhas_out)
    if df_rel.empty:
        return pd.DataFrame(), pd.DataFrame()

    ordem_modo = {"Leve": 1, "Moderado": 2, "Extremo": 3}
    df_rel["__ord_modo"] = df_rel["Modo operação"].map(ordem_modo).fillna(999)
    df_rel = df_rel.sort_values(["Código", "Hectare", "__ord_modo"]).reset_index(drop=True)
    df_rel = df_rel.drop(columns=["__ord_modo"])

    df_export = df_rel[["Modelo", "Modo operação", "Código", "R(t)", "Hectare", "Eta", "Beta"]].copy()
    return df_rel, df_export


def gerar_excel_confiabilidade(df_export):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df_export.to_excel(writer, index=False, sheet_name="Confiabilidade")
    buffer.seek(0)
    return buffer

# ------------------------------------------------------------
# Assinatura (para evitar reset ao navegar) + Reprocessamento central
# ------------------------------------------------------------
def _assinatura_atual():
    mults = st.session_state.get("multiplicadores_operacao",
                                 {"Leve": 1.30, "Moderado": 1.00, "Extremo": 0.70})
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
        st.session_state.get("considerar_anos"),
        float(mults.get("Leve", 1.30)),
        float(mults.get("Moderado", 1.00)),
        float(mults.get("Extremo", 0.70)),
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

        if show_msg:
            st.success("Dados reprocessados com base nos parâmetros atuais.")
    else:
        if show_msg:
            st.info("Parâmetros não mudaram. Mantendo cálculos e ajustes atuais.")


# ------------------------------------------------------------
# Layout principal (páginas)
# ------------------------------------------------------------
init_session_state()

st.sidebar.image(
    "https://i.postimg.cc/Kz9xcnJr/Chat-GPT-Image-6-de-fev-de-2026-15-14-23.png",
    use_container_width=True
)

pagina = st.sidebar.radio(
    "Navegação",
    [
        "1. Entrada de Dados",
        "2. Ajustes de Peças",
        "3. Resumo / Resultados",
        "4. Análise operacional",
        "5. Plano de Manutenção",
        "6. Confiabilidade"
    ]
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
                prev_pecas["Hectare/Proporção"] = prev_pecas["Hectare/Proporção"].apply(format_thousand_no_decimals)
            if "Proporção de troca (%)" in prev_pecas.columns:
                prev_pecas["Proporção de troca (%)"] = (
                    pd.to_numeric(prev_pecas["Proporção de troca (%)"], errors="coerce")
                    .fillna(0)
                    .clip(0, 100)
                    .astype(int)
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
                                            value=float(mults.get("Leve", 1.30)))
        with cB:
            mults["Moderado"] = st.number_input("Multiplicador - Moderado",
                                                min_value=0.1, max_value=5.0, step=0.1,
                                                value=float(mults.get("Moderado", 1.00)))
        with cC:
            mults["Extremo"] = st.number_input("Multiplicador - Extremo",
                                               min_value=0.1, max_value=5.0, step=0.1,
                                               value=float(mults.get("Extremo", 0.70)))
        st.session_state["multiplicadores_operacao"] = mults
        st.caption("Os multiplicadores acima são aplicados sobre o Hectare/Proporção de cada peça (base).")

        st.session_state["default_proporcao_troca"] = st.slider(
            "Proporção de troca padrão (%) (fallback quando a peça não traz na planilha)",
            min_value=0, max_value=100, step=1,
            value=int(st.session_state["default_proporcao_troca"])
        )
        st.caption("Se a planilha de peças tiver a coluna 'Proporção de troca (%)', ela prevalece como padrão por peça.")

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
    st.subheader("Horizonte de cálculo (vida útil das máquinas)")
    opcoes_horizonte = [
        "Considerar todas as máquinas como novas",
        "Considerar com base no ano e estado"
    ]
    valor_atual_horizonte = st.session_state.get("considerar_anos", "Considerar todas as máquinas como novas")
    idx_horizonte = opcoes_horizonte.index(valor_atual_horizonte) if valor_atual_horizonte in opcoes_horizonte else 0
    st.session_state["considerar_anos"] = st.radio(
        "Como considerar a vida útil das máquinas?",
        opcoes_horizonte,
        horizontal=True,
        index=idx_horizonte
    )

    st.caption(
        "- **Considerar todas as máquinas como novas**: ignora histórico de uso e considera 1 ano de operação para todas.\n"
        "- **Considerar com base no ano e estado**: usa o ano da máquina e o estado (Usado/Novo) para calcular o número de anos de uso."
    )

    st.markdown("---")
    st.caption("⚙️ Regra do sistema: o cálculo de quantidade de peças é sempre feito no modo **Inteiro** (ciclos completos).")

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
        st.write(
            f"Modelo: {resumo_ref.get('modelo','-')} • Chassi ref: {resumo_ref.get('chassi_ref','-')} "
            f"• Linhas: {resumo_ref.get('linhas_maquina','?')} • Frota (modelo): {resumo_ref.get('n_chassis_frota', 1)}"
        )

        st.write("Edite os parâmetros peça a peça. Esses ajustes alimentam os cálculos finais.")
        # ✅ REMOVIDO: mensagem sobre persistência ao alternar páginas

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

        for _, row in df_unique.iterrows():
            codigo_item = format_codigo(row["Código"])
            st.markdown("---")
            st.subheader(f"{codigo_item} - {row['Descrição']}")

            base_hect = float(row["hectare_proporcao_efetivo"])
            base_prop = int(round(float(row["proporcao_troca_%"])))

            aj = ajustes.get(codigo_item, {})

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
                # ✅ REMOVIDO: "⚙️ Regra do sistema..." dentro da página 2

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

                _ = st.number_input(
                    "Hectare/Proporção",
                    min_value=0.0,
                    step=1.0,
                    value=float(st.session_state[key_hect]),
                    key=key_hect,
                    on_change=lambda kh=key_hect, kr=key_ref, t=tipo_prop_lower, nl=n_linhas_ref: cb_from_hect(kh, kr, t, nl)
                )

                _ = st.number_input(
                    "Hectare referência",
                    min_value=0.0,
                    step=1.0,
                    value=float(st.session_state[key_ref]),
                    key=key_ref,
                    on_change=lambda kh=key_hect, kr=key_ref, t=tipo_prop_lower, nl=n_linhas_ref: cb_from_ref(kh, kr, t, nl)
                )

                _ = st.slider(
                    "Proporção de troca (%)",
                    min_value=0, max_value=100,
                    value=int(st.session_state[key_prop]),
                    key=key_prop
                )

                synced_hect = float(st.session_state[key_hect])

                vida_total, qtd_prevista, qtd_ciclos = calcular_hect_ref_e_qtd_prevista(
                    row,
                    resumo_ref,
                    synced_hect,
                    int(st.session_state[key_prop]),
                )
                st.write(f"**Quantidade prevista**: {int(round(qtd_prevista))}")
                st.write(f"**Quantidade de ciclos:** {int(np.floor(qtd_ciclos))}")

            with cC:
                st.write(f"Proporção declarada: {row['Proporção']}")
                st.write(f"Qtd/Proporção: {row['Qtd/Proporção']}")
                # ✅ REMOVIDO: "Hectare/Proporção (original): ..." (texto + valor)
                # ✅ REMOVIDO: "Proporção de troca (base): ..." (texto + valor)
                st.write(f"Linhas do chassi (ref): {n_linhas_ref}")

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

            antigo = ajustes.get(codigo_item, {})

            ajustes[codigo_item] = {
                "hect": float(synced_hect) if manual_hect else antigo.get("hect"),
                "prop": int(st.session_state[key_prop]) if manual_prop else antigo.get("prop"),
                "manual_hect": manual_hect or antigo.get("manual_hect", False),
                "manual_prop": manual_prop or antigo.get("manual_prop", False),

                # mantém possíveis bases importadas
                "hect_base": antigo.get("hect_base"),
                "manual_hect_base": antigo.get("manual_hect_base", False),
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

        st.session_state["df_pecas_proc"] = recalcular_pecas_pos_ajuste(st.session_state["df_pecas_proc"], resumo_ref)

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

        # ✅ TABELA: informações do(s) chassi(s) conforme seleção da página 1 (SEM anualização)
        st.markdown("---")
        st.subheader("Informações da máquina (conforme seleção da Página 1)")

        df_maqs = st.session_state.get("df_maquinas_raw")
        df_maqs_proc = st.session_state.get("df_maquinas_proc")
        modelo_sel = st.session_state.get("modelo_selecionado")
        chassi_sel = st.session_state.get("chassi_selecionado")

        info_rows = []
        if df_maqs is not None and not df_maqs.empty and modelo_sel:
            tmp = df_maqs.copy()
            tmp = tmp[tmp["Modelo"] == modelo_sel].copy()
            if "Chassi" in tmp.columns:
                tmp["Chassi"] = tmp["Chassi"].astype(str)
            if chassi_sel and chassi_sel != "Todos":
                tmp = tmp[tmp["Chassi"].astype(str) == str(chassi_sel)].copy()

            # tenta trazer anos_uso da base processada (quando existir)
            anos_uso_map = {}
            if df_maqs_proc is not None and not df_maqs_proc.empty and "Chassi" in df_maqs_proc.columns:
                tp = df_maqs_proc.copy()
                tp["Chassi"] = tp["Chassi"].astype(str)
                if "anos_uso" in tp.columns:
                    anos_uso_map = dict(zip(tp["Chassi"], tp["anos_uso"]))

            for _, r in tmp.iterrows():
                ch = str(r.get("Chassi", ""))
                info_rows.append({
                    "Chassi": ch,
                    "Linhas": int(r.get("Linhas", 0) or 0) if pd.notna(r.get("Linhas", np.nan)) else 0,
                    "Espaçamento": float(r.get("Espaçamento", 0.0) or 0.0) if pd.notna(r.get("Espaçamento", np.nan)) else 0.0,
                    "Ano": format_ano(r.get("Ano", "")),
                    "Estado": str(r.get("Estado", "Novo")).strip().capitalize() if "Estado" in tmp.columns else "Novo",
                    "Anos de uso (calculado)": int(anos_uso_map.get(ch, 1) or 1)
                })

        df_info = pd.DataFrame(info_rows)
        if df_info.empty:
            st.info("Sem dados suficientes para exibir as informações da máquina.")
        else:
            st.dataframe(df_info, use_container_width=True, hide_index=True)

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
            val_hect = format_currency(indicadores['custo_medio_por_hectare']) if not np.isnan(indicadores['custo_medio_por_hectare']) else "n/d"
            st.metric("Custo médio por hectare (R$/ha)", value=val_hect)
            st.caption("Base: escopo selecionado.")
        with col_r3:
            val_hora = format_currency(indicadores['custo_medio_por_hora']) if not np.isnan(indicadores['custo_medio_por_hora']) else "n/d"
            st.metric("Custo médio por hora (R$/h)", value=val_hora)
            st.caption("Base: escopo selecionado.")

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

        # ✅ IMPORTANTE: NÃO anualizar (removido pedido de coluna "Ano")
        df_export_preview_num = agregar_para_exportacao(
            st.session_state["df_pecas_proc"],
            resumo_ref,
            familia_filter=st.session_state["filtro_familia_resumo"],
            escopo=st.session_state["escopo_resumo"]
        ).copy()

        if not df_export_preview_num.empty:
            df_export_preview_num = df_export_preview_num[
                pd.to_numeric(df_export_preview_num["Qtd recomendada"], errors="coerce").fillna(0.0) > 0
            ].copy()

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
            df_export_preview_num["Qtd recomendada"] = df_export_preview_num["Qtd recomendada"].apply(
                lambda x: int(round(x if pd.notna(x) else 0))
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
            label="⬇️ Exportar",
            data=buffer_xlsx,
            file_name="planejamento_manutencao.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# ------------------------------------------------------------
# PÁGINA 4 - Análise operacional (mantida como estava; rótulos já existentes)
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

        col_p1, col_p2 = st.columns(2)

        with col_p1:
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
            if "inicio_operacao" not in st.session_state:
                st.session_state["inicio_operacao"] = datetime.date.today()
            inicio_operacao = st.date_input(
                "Início da operação",
                value=st.session_state["inicio_operacao"],
                key="inicio_operacao_date"
            )
            st.session_state["inicio_operacao"] = inicio_operacao

        hectare_ano_ref = float(resumo_ref.get("ha_ano_maquina", 0.0) or 0.0)
        hectare_hora_ref = float(resumo_ref.get("ha_hora_maquina", 0.0) or 0.0)

        horas_total_operacao = hectare_ano_ref / hectare_hora_ref if hectare_ano_ref > 0 and hectare_hora_ref > 0 else 0.0
        total_dias = horas_total_operacao / horas_trabalho_dia if horas_total_operacao > 0 and horas_trabalho_dia > 0 else 0.0

        fim_operacao = inicio_operacao + datetime.timedelta(days=int(round(total_dias))) if isinstance(inicio_operacao, datetime.date) else None
        hectare_por_dia_ref = hectare_hora_ref * horas_trabalho_dia

        if hectare_ano_ref <= 0 or hectare_hora_ref <= 0:
            st.warning("Verifique os parâmetros da página 1 (Hectare médio por ano, Hectares por hora e largura).")

        st.markdown("---")

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
            df_maqs_local = df_maqs_all.copy()

        if df_maqs_local.empty:
            st.warning("Não há chassis disponíveis para o escopo selecionado.")
        else:
            if escopo_label == "Chassi específico":
                hectare_por_dia = hectare_por_dia_ref
            else:
                hectare_por_dia = 0.0
                for _, m in df_maqs_local.iterrows():
                    ha_hora_maq = float(m.get("ha_hora_chassi", 0.0) or 0.0)
                    hectare_por_dia += ha_hora_maq * horas_trabalho_dia

            hectare_total = total_dias * hectare_por_dia

            st.markdown("### Resumo operacional (máquina de referência)")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Horas total de operação", f"{horas_total_operacao:.2f}")
            with c2:
                st.metric("Total de dias", f"{total_dias:.2f}")
            with c3:
                st.metric("Hectare por dia", f"{hectare_por_dia:.2f}")
            with c4:
                st.metric("Hectare total", f"{hectare_total:.2f}")

            col_d1, col_d2 = st.columns(2)
            inicio_str = inicio_operacao.strftime("%d/%m/%Y") if isinstance(inicio_operacao, datetime.date) else "-"
            fim_str = fim_operacao.strftime("%d/%m/%Y") if isinstance(fim_operacao, datetime.date) else "-"
            with col_d1:
                st.write(f"**Início da operação:** {inicio_str}")
            with col_d2:
                st.write(f"**Fim da operação (ref.):** {fim_str}")

            if horas_trabalho_dia <= 0:
                st.warning("Informe um valor positivo em 'Horas de trabalho por dia' para gerar o calendário.")
            else:
                linhas_calendario = []
                considerar_anos_flag = st.session_state.get("considerar_anos", "Considerar todas as máquinas como novas")

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

                    start_prev_ha = (anos_uso_maq - 1) * ha_ano_maq if (considerar_anos_flag == "Considerar com base no ano e estado" and anos_uso_maq > 1) else 0.0
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

                        if tipo_prop == "linha":
                            vida_total_ha = vida_base * n_linhas_maq
                            qtd_ciclo_teorico = qtd_por_prop * n_linhas_maq
                        else:
                            vida_total_ha = vida_base
                            qtd_ciclo_teorico = qtd_por_prop

                        if vida_total_ha <= 0 or qtd_ciclo_teorico <= 0:
                            continue

                        qtd_recomendada_maq = _quantidade_para_maquina_especifica(row_p, ha_ano_maq, n_linhas_maq, anos_uso_maq)
                        if qtd_recomendada_maq <= 0:
                            continue

                        q_evento_cheio = qtd_ciclo_teorico * (prop_troca / 100.0)

                        if q_evento_cheio <= 0:
                            quantidades = [qtd_recomendada_maq]
                            offsets_ha = [0.0]
                        else:
                            k_start = int(np.floor(start_prev_ha / vida_total_ha)) + 1
                            k_end = int(np.floor(end_current_ha / vida_total_ha))
                            full_ks = list(range(k_start, k_end + 1)) if k_end >= k_start else []

                            quantidades, offsets_ha = [], []
                            for k in full_ks:
                                A_k = k * vida_total_ha
                                offset_ha = A_k - start_prev_ha
                                offsets_ha.append(offset_ha)
                                quantidades.append(q_evento_cheio)

                            total_full = sum(quantidades) if quantidades else 0.0
                            resto = float(qtd_recomendada_maq - total_full)

                            if resto > 1e-6:
                                quantidades.append(resto)
                                offsets_ha.append(end_current_ha - start_prev_ha)

                            if not quantidades:
                                quantidades = [qtd_recomendada_maq]
                                offsets_ha = [end_current_ha - start_prev_ha]

                        for q_evt, off_ha in zip(quantidades, offsets_ha):
                            if isinstance(inicio_operacao, datetime.date):
                                dias_offset = off_ha / hectare_por_dia_maq if hectare_por_dia_maq > 0 else 0.0
                                data_evt = inicio_operacao + datetime.timedelta(days=int(round(dias_offset)))
                                data_troca_str = data_evt.strftime("%m/%Y")
                            else:
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

                df_cal = pd.DataFrame(linhas_calendario)

                if not df_cal.empty:
                    group_cols = ["Família", "Código", "Descrição", "Data troca"]
                    df_cal = df_cal.groupby(group_cols, as_index=False).agg({"Quantidade peça": "sum", "Custo": "sum"})

                    df_cal["Data troca"] = pd.to_datetime("01/" + df_cal["Data troca"].astype(str), format="%d/%m/%Y", errors="coerce")
                    df_cal = df_cal.dropna(subset=["Data troca"])
                    df_cal = df_cal.sort_values("Data troca").reset_index(drop=True)

                    df_cal["Quantidade peça"] = df_cal["Quantidade peça"].round(0)

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
                            index=(familias_dropdown_p4.index(st.session_state["filtro_familia_p4"])
                                   if st.session_state["filtro_familia_p4"] in familias_dropdown_p4 else 0)
                        )
                    with col_f2:
                        st.session_state["filtro_campo_p4"] = st.selectbox(
                            "Filtrar por campo",
                            ["Todos", "Código", "Descrição", "Data troca"],
                            index=(["Todos", "Código", "Descrição", "Data troca"].index(st.session_state["filtro_campo_p4"])
                                   if st.session_state["filtro_campo_p4"] in ["Todos", "Código", "Descrição", "Data troca"] else 0)
                        )
                    with col_f3:
                        st.session_state["filtro_valor_p4"] = st.text_input(
                            "Valor do filtro (contém)",
                            value=st.session_state["filtro_valor_p4"]
                        )

                    fam_sel_p4 = st.session_state["filtro_familia_p4"]
                    if fam_sel_p4 != "Todos":
                        df_cal = df_cal[df_cal["Família"] == fam_sel_p4]

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
                        else:
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

                    if not df_cal.empty:
                        df_export_p4 = df_cal.copy()
                        buffer_p4 = BytesIO()
                        with pd.ExcelWriter(buffer_p4, engine="xlsxwriter") as writer:
                            df_export_p4.to_excel(writer, index=False, sheet_name="Analise_operacional")
                        buffer_p4.seek(0)

                        st.download_button(
                            label="⬇️ Exportar",
                            data=buffer_p4,
                            file_name="analise_operacional.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )

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

                        if "Data troca" not in df_cal.columns:
                            st.error("A coluna 'Data troca' não foi encontrada para gerar o gráfico.")
                        else:
                            if not np.issubdtype(df_cal["Data troca"].dtype, np.datetime64):
                                df_cal["Data troca"] = pd.to_datetime(df_cal["Data troca"], errors="coerce")

                            df_chart_base = df_cal.dropna(subset=["Data troca"]).copy()

                            if df_chart_base.empty:
                                st.info("Sem dados válidos de 'Data troca' para montar o gráfico (após filtros).")
                            else:
                                chart_month = (
                                    df_chart_base
                                    .set_index("Data troca")[["Quantidade peça", "Custo"]]
                                    .resample("MS")
                                    .sum()
                                    .reset_index()
                                )

                                if chart_month.empty:
                                    st.info("Nenhum dado mensal disponível para o gráfico com os filtros atuais.")
                                else:
                                    chart_month["DataLabel"] = chart_month["Data troca"].dt.strftime("%m/%Y")
                                    chart_month["Quantidade_str"] = (
                                        pd.to_numeric(chart_month["Quantidade peça"], errors="coerce")
                                        .fillna(0)
                                        .round(0)
                                        .astype(int)
                                        .astype(str)
                                    )
                                    chart_month["Custo_str"] = (
                                        pd.to_numeric(chart_month["Custo"], errors="coerce")
                                        .fillna(0.0)
                                        .apply(format_currency)
                                    )

                                    chart = (
                                        alt.Chart(chart_month)
                                        .mark_bar(color="#A70623")
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

                                    label_field = "Quantidade_str:N" if metrica_graf == "Quantidade de peças" else "Custo_str:N"
                                    labels = (
                                        alt.Chart(chart_month)
                                        .mark_text(dy=-6, color="black", fontSize=11)
                                        .encode(
                                            x=alt.X("DataLabel:N", sort=alt.SortField(field="Data troca", order="ascending")),
                                            y=alt.Y(f"{y_col}:Q"),
                                            text=alt.Text(label_field)
                                        )
                                    )

                                    st.altair_chart(chart + labels, use_container_width=True)
                else:
                    st.info("Nenhum evento de troca foi gerado com os parâmetros atuais.")


# ------------------------------------------------------------
# PÁGINA 5 - Plano de Manutenção (com botão XLSX do gráfico - 3 colunas)
# ------------------------------------------------------------
elif pagina == "5. Plano de Manutenção":
    st.title("5. Plano de Manutenção")

    run_processamento_if_needed(show_msg=False)

    if (
        st.session_state["df_pecas_proc"] is None
        or st.session_state["df_maquinas_proc"] is None
        or st.session_state["df_pecas_proc"].empty
        or st.session_state["df_maquinas_proc"].empty
    ):
        st.warning("Você ainda não carregou dados ou não processou na página 1. Volte para '1. Entrada de Dados'.")
    else:
        df_maqs = st.session_state["df_maquinas_proc"].copy()
        df_maqs["Chassi"] = df_maqs["Chassi"].astype(str)

        chassis_opcoes_p5 = (
            df_maqs["Chassi"].astype(str).sort_values().unique().tolist()
            if "Chassi" in df_maqs.columns else []
        )
        if not chassis_opcoes_p5:
            st.warning("Não há chassis disponíveis para o modelo processado.")
        else:
            if st.session_state.get("plano_chassi_selecionado") not in chassis_opcoes_p5:
                st.session_state["plano_chassi_selecionado"] = chassis_opcoes_p5[0]

            st.session_state["plano_chassi_selecionado"] = st.selectbox(
                "Selecione o chassi para o Plano de Manutenção (independente da página 1)",
                chassis_opcoes_p5,
                index=chassis_opcoes_p5.index(st.session_state["plano_chassi_selecionado"])
            )

            chassi_p5 = str(st.session_state["plano_chassi_selecionado"])

            row_maq = df_maqs[df_maqs["Chassi"] == chassi_p5]
            if row_maq.empty:
                st.warning("Chassi não encontrado no processamento atual.")
            else:
                row_maq = row_maq.iloc[0]

                df_raw = st.session_state.get("df_maquinas_raw")
                if df_raw is not None and not df_raw.empty and "Chassi" in df_raw.columns:
                    tmp_raw = df_raw.copy()
                    tmp_raw["Chassi"] = tmp_raw["Chassi"].astype(str)
                    row_raw = tmp_raw[tmp_raw["Chassi"].astype(str) == chassi_p5]
                    row_raw = row_raw.iloc[0] if not row_raw.empty else None
                else:
                    row_raw = None

                modelo_show = (row_raw.get("Modelo") if row_raw is not None else row_maq.get("Modelo"))
                linhas_show = (row_raw.get("Linhas") if row_raw is not None else row_maq.get("Linhas"))
                esp_show = (row_raw.get("Espaçamento") if row_raw is not None else row_maq.get("Espaçamento"))
                ano_bruto = (row_raw.get("Ano") if row_raw is not None else row_maq.get("Ano"))
                ano_show = format_ano(ano_bruto)

                df_info = pd.DataFrame([{
                    "Modelo": modelo_show,
                    "Linhas": linhas_show,
                    "Espaçamento": esp_show,
                    "Ano": ano_show
                }])

                st.subheader("Informações do chassi selecionado")
                st.dataframe(df_info, use_container_width=True, hide_index=True)

                st.markdown("---")
                st.session_state["plano_tempo_operacao_anos"] = st.number_input(
                    "Tempo de operação (anos)",
                    min_value=1,
                    step=1,
                    value=int(st.session_state.get("plano_tempo_operacao_anos", 1)),
                    help="Número inteiro de anos (ciclos) para simular: Ano 1, Ano 2, ...",
                )
                tempo_anos = int(st.session_state["plano_tempo_operacao_anos"])

                ha_ano_maq = float(row_maq.get("ha_ano_chassi", 0.0) or 0.0)
                ha_hora_maq = float(row_maq.get("ha_hora_chassi", 0.0) or 0.0)
                n_linhas_maq = int(row_maq.get("Linhas", 1) or 1)

                if ha_ano_maq <= 0:
                    st.warning("Hectare/ano desse chassi ficou 0. Verifique os parâmetros da página 1.")
                else:
                    # ----------------------------
                    # Monta plano por ano/ciclo
                    # ----------------------------
                    df_pecas_base = st.session_state["df_pecas_proc"].copy()
                    df_unique_p = df_pecas_base.groupby("Código").first().reset_index()
                    df_unique_p["Código"] = df_unique_p["Código"].apply(format_codigo)

                    linhas_out = []
                    for ano_ciclo in range(1, tempo_anos + 1):
                        ano_label = f"Ano {ano_ciclo}"
                        for _, p in df_unique_p.iterrows():
                            qtd = _quantidade_para_maquina_especifica_plano(
                                p, ha_ano_maq, n_linhas_maq, ano_ciclo
                            )
                            if qtd <= 0:
                                continue

                            custo_unit = float(p.get("custo_unitario", 0.0) or 0.0)
                            custo_total = float(qtd) * custo_unit

                            linhas_out.append({
                                "Ano": ano_label,
                                "Família": p.get("Família", ""),
                                "Código": format_codigo(p.get("Código", "")),
                                "Descrição": p.get("Descrição", ""),
                                "Qtd recomendada": float(qtd),
                                "Custo total (R$)": float(custo_total),
                            })

                    df_plano = pd.DataFrame(linhas_out)

                    # ----------------------------
                    # Indicadores (similar ao Resumo/Resultados)
                    # ----------------------------
                    custo_total_periodo = float(df_plano["Custo total (R$)"].sum()) if not df_plano.empty else 0.0
                    total_ha_periodo = float(ha_ano_maq) * float(tempo_anos)

                    horas_ano = (float(ha_ano_maq) / float(ha_hora_maq)) if (ha_hora_maq and ha_hora_maq > 0) else 0.0
                    total_horas_periodo = float(horas_ano) * float(tempo_anos)

                    custo_medio_por_ha = (custo_total_periodo / total_ha_periodo) if total_ha_periodo > 0 else np.nan
                    custo_medio_por_hora = (custo_total_periodo / total_horas_periodo) if total_horas_periodo > 0 else np.nan

                    st.markdown("### Indicadores do Plano (período simulado)")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Custo total sugerido (período)", value=format_currency(custo_total_periodo))
                        st.caption(f"Base: {tempo_anos} ano(s) • chassi {chassi_p5}.")
                    with c2:
                        val_hect = format_currency(custo_medio_por_ha) if not np.isnan(custo_medio_por_ha) else "n/d"
                        st.metric("Custo médio por hectare (R$/ha)", value=val_hect)
                        st.caption(f"Base: ha/ano ({ha_ano_maq:.2f}) × {tempo_anos}.")
                    with c3:
                        val_hora = format_currency(custo_medio_por_hora) if not np.isnan(custo_medio_por_hora) else "n/d"
                        st.metric("Custo médio por hora (R$/h)", value=val_hora)
                        base_h = f"{horas_ano:.2f}" if horas_ano > 0 else "n/d"
                        st.caption(f"Base: horas/ano ({base_h}) × {tempo_anos}.")

                    st.markdown("---")
                    st.subheader("Plano de Manutenção (por ano/ciclo)")

                    if df_plano.empty:
                        st.info("Nenhum item com quantidade recomendada > 0 para o chassi e período selecionados.")
                    else:
                        df_plano["Qtd recomendada"] = (
                            pd.to_numeric(df_plano["Qtd recomendada"], errors="coerce")
                            .fillna(0.0)
                            .round(0)
                        )

                        familias_p5 = sorted(df_plano["Família"].dropna().unique().tolist())
                        familias_dropdown_p5 = ["Todos"] + familias_p5

                        col_f1, col_f2, col_f3 = st.columns([1.2, 1.2, 2])
                        with col_f1:
                            st.session_state["filtro_familia_p5"] = st.selectbox(
                                "Família",
                                familias_dropdown_p5,
                                index=(
                                    familias_dropdown_p5.index(st.session_state["filtro_familia_p5"])
                                    if st.session_state["filtro_familia_p5"] in familias_dropdown_p5
                                    else 0
                                )
                            )
                        with col_f2:
                            st.session_state["filtro_campo_p5"] = st.selectbox(
                                "Filtrar por campo",
                                ["Todos", "Ano", "Código", "Descrição", "Família"],
                                index=(
                                    ["Todos", "Ano", "Código", "Descrição", "Família"].index(st.session_state["filtro_campo_p5"])
                                    if st.session_state["filtro_campo_p5"] in ["Todos", "Ano", "Código", "Descrição", "Família"]
                                    else 0
                                )
                            )
                        with col_f3:
                            st.session_state["filtro_valor_p5"] = st.text_input(
                                "Valor do filtro (contém)",
                                value=st.session_state["filtro_valor_p5"]
                            )

                        fam_sel = st.session_state["filtro_familia_p5"]
                        if fam_sel != "Todos":
                            df_plano = df_plano[df_plano["Família"] == fam_sel]

                        filtro_txt = st.session_state["filtro_valor_p5"].strip().lower()
                        campo = st.session_state["filtro_campo_p5"]

                        if filtro_txt:
                            if campo == "Todos":
                                mask = (
                                    df_plano["Ano"].astype(str).str.lower().str.contains(filtro_txt)
                                    | df_plano["Código"].astype(str).str.lower().str.contains(filtro_txt)
                                    | df_plano["Descrição"].astype(str).str.lower().str.contains(filtro_txt)
                                    | df_plano["Família"].astype(str).str.lower().str.contains(filtro_txt)
                                )
                            elif campo == "Ano":
                                mask = df_plano["Ano"].astype(str).str.lower().str.contains(filtro_txt)
                            elif campo == "Código":
                                mask = df_plano["Código"].astype(str).str.lower().str.contains(filtro_txt)
                            elif campo == "Descrição":
                                mask = df_plano["Descrição"].astype(str).str.lower().str.contains(filtro_txt)
                            else:
                                mask = df_plano["Família"].astype(str).str.lower().str.contains(filtro_txt)
                            df_plano = df_plano[mask]

                        st.dataframe(
                            df_plano,
                            column_config={
                                "Ano": st.column_config.TextColumn("Ano"),
                                "Família": st.column_config.TextColumn("Família"),
                                "Código": st.column_config.TextColumn("Código"),
                                "Descrição": st.column_config.TextColumn("Descrição"),
                                "Qtd recomendada": st.column_config.NumberColumn("Qtd recomendada", format="%.0f"),
                                "Custo total (R$)": st.column_config.NumberColumn("Custo total (R$)", format="R$ %.2f"),
                            },
                            hide_index=True,
                            use_container_width=True,
                        )

                        buffer_p5 = BytesIO()
                        with pd.ExcelWriter(buffer_p5, engine="xlsxwriter") as writer:
                            df_plano.to_excel(writer, index=False, sheet_name="Plano_manutencao")
                        buffer_p5.seek(0)

                        st.download_button(
                            label="⬇️ Exportar",
                            data=buffer_p5,
                            file_name="plano_manutencao.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )

                        st.markdown("---")
                        st.subheader("Gráfico por ano/ciclo")

                        metrica = st.radio(
                            "Métrica do gráfico",
                            ["Custo total (R$)", "Custo por hectare (R$/ha)"],
                            horizontal=True,
                            index=(0 if st.session_state.get("metrica_graf_p5", "Custo total (R$)") == "Custo total (R$)" else 1),
                        )
                        st.session_state["metrica_graf_p5"] = metrica

                        df_g = df_plano.groupby("Ano", as_index=False).agg({"Custo total (R$)": "sum"})
                        df_g["Custo por hectare (R$/ha)"] = df_g["Custo total (R$)"] / float(ha_ano_maq)

                        if metrica == "Custo total (R$)":
                            y_col = "Custo total (R$)"
                            y_title = "Custo total (R$)"
                        else:
                            y_col = "Custo por hectare (R$/ha)"
                            y_title = "Custo por hectare (R$/ha)"

                        def _ano_num(x):
                            try:
                                return int(str(x).lower().replace("ano", "").strip())
                            except Exception:
                                return 999999

                        df_g["Ano_num"] = df_g["Ano"].apply(_ano_num)
                        df_g = df_g.sort_values("Ano_num").reset_index(drop=True)

                        # --------- ✅ EXPORTAR DADOS DO GRÁFICO EM XLSX (APENAS 3 COLUNAS) ---------
                        df_g_export = df_g[["Ano", "Custo total (R$)", "Custo por hectare (R$/ha)"]].copy()

                        buffer_graf_p5 = BytesIO()
                        with pd.ExcelWriter(buffer_graf_p5, engine="xlsxwriter") as writer:
                            df_g_export.to_excel(writer, index=False, sheet_name="Dados_grafico")
                        buffer_graf_p5.seek(0)

                        st.download_button(
                            label="⬇️ Exportar",
                            data=buffer_graf_p5,
                            file_name="dados_grafico_plano_manutencao.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                        st.caption("Obs.: o menu do gráfico exporta CSV; para Excel (.xlsx), use o botão acima.")

                        # --------- GRÁFICO ---------
                        df_g["CustoTotal_str"] = df_g["Custo total (R$)"].apply(format_currency)
                        df_g["CustoHa_str"] = df_g["Custo por hectare (R$/ha)"].apply(format_currency)

                        chart = (
                            alt.Chart(df_g)
                            .mark_bar(color="#A70623")
                            .encode(
                                x=alt.X("Ano:N", title="Ano do ciclo", sort=alt.SortField(field="Ano_num", order="ascending")),
                                y=alt.Y(f"{y_col}:Q", title=y_title),
                                tooltip=[
                                    alt.Tooltip("Ano:N", title="Ano"),
                                    alt.Tooltip("CustoTotal_str:N", title="Custo total"),
                                    alt.Tooltip("CustoHa_str:N", title="Custo por hectare"),
                                ]
                            )
                            .interactive()
                        )

                        # RÓTULOS (DATA LABELS) conforme seleção
                        label_field = "CustoTotal_str:N" if metrica == "Custo total (R$)" else "CustoHa_str:N"
                        labels = (
                            alt.Chart(df_g)
                            .mark_text(dy=-6, color="black", fontSize=12)
                            .encode(
                                x=alt.X("Ano:N", sort=alt.SortField(field="Ano_num", order="ascending")),
                                y=alt.Y(f"{y_col}:Q"),
                                text=alt.Text(label_field)
                            )
                        )

                        st.altair_chart(chart + labels, use_container_width=True)

                        st.markdown("---")

                        buffer_bi = gerar_excel_bi_plano_manutencao(
                            chassi_p5=chassi_p5,
                            tempo_anos=tempo_anos
                        )

                        st.download_button(
                            label="Dados BI",
                            data=buffer_bi,
                            file_name="dados_bi_plano_manutencao.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )

# ------------------------------------------------------------
# PÁGINA 6 - Confiabilidade
# ------------------------------------------------------------
elif pagina == "6. Confiabilidade":
    st.title("6. Confiabilidade")

    run_processamento_if_needed(show_msg=False)

    if (
        st.session_state["df_pecas_raw"] is None
        or st.session_state["df_maquinas_raw"] is None
        or st.session_state["modelo_selecionado"] is None
        or st.session_state["resumo_maquina_ref"] is None
    ):
        st.warning("Primeiro carregue os dados e processe a Página 1.")
    else:
        st.subheader("Importar tabela de Beta (.xlsx)")
        beta_file = st.file_uploader(
            "Planilha Beta com colunas: Código, Família, Descrição, Beta",
            type=["xlsx"],
            key="upload_beta_xlsx"
        )

        if beta_file is not None:
            try:
                st.session_state["df_beta_raw"] = pd.read_excel(beta_file)
                st.session_state["beta_import_filename"] = beta_file.name
            except Exception as e:
                st.error(f"Falha ao ler a planilha Beta: {e}")
                st.session_state["df_beta_raw"] = None

        df_beta_raw = st.session_state.get("df_beta_raw")

        if df_beta_raw is None or df_beta_raw.empty:
            st.info("Importe a planilha de Beta para habilitar o gráfico e a exportação.")
        else:
            try:
                df_beta = higienizar_beta(df_beta_raw)
            except Exception as e:
                st.error(f"Erro na planilha Beta: {e}")
                st.stop()

            df_rel, df_export = montar_base_confiabilidade(df_beta)

            if df_rel.empty:
                st.warning("Nenhum código da planilha Beta encontrou correspondência na base de peças do modelo selecionado.")
            else:
                st.success(f"Base de confiabilidade gerada com {len(df_export):,} linhas.".replace(",", "."))

                # ---------
                # Filtro por código
                # ---------
                codigos_disp = (
                    df_rel[["Código", "Descrição"]]
                    .drop_duplicates()
                    .sort_values(["Código", "Descrição"])
                    .reset_index(drop=True)
                )

                opcoes_codigo = codigos_disp["Código"].tolist()

                if st.session_state.get("codigo_confiabilidade_sel") not in opcoes_codigo:
                    st.session_state["codigo_confiabilidade_sel"] = opcoes_codigo[0]

                codigo_sel = st.selectbox(
                    "Selecione o código para visualizar a curva R(t)",
                    opcoes_codigo,
                    index=opcoes_codigo.index(st.session_state["codigo_confiabilidade_sel"])
                )
                st.session_state["codigo_confiabilidade_sel"] = codigo_sel

                df_plot = df_rel[df_rel["Código"] == codigo_sel].copy()

                descricao_sel = ""
                if not df_plot.empty and "Descrição" in df_plot.columns:
                    descricao_sel = str(df_plot["Descrição"].dropna().iloc[0]) if not df_plot["Descrição"].dropna().empty else ""

                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"**Código:** {codigo_sel}")
                with c2:
                    st.write(f"**Descrição:** {descricao_sel if descricao_sel else '-'}")

                # ---------
                # Gráfico R(t)
                # ---------
                st.markdown("### Curva de confiabilidade R(t)")

                ordem_modo = ["Leve", "Moderado", "Extremo"]
                df_plot["Modo operação"] = pd.Categorical(df_plot["Modo operação"], categories=ordem_modo, ordered=True)
                df_plot = df_plot.sort_values(["Modo operação", "Hectare"]).reset_index(drop=True)

                graf = (
                    alt.Chart(df_plot)
                    .mark_line()
                    .encode(
                        x=alt.X("Hectare:Q", title="Hectare"),
                        y=alt.Y("R(t):Q", title="Confiabilidade R(t)", scale=alt.Scale(domain=[0, 1.05])),
                        color=alt.Color("Modo operação:N", title="Modo operação"),
                        tooltip=[
                            alt.Tooltip("Código:N", title="Código"),
                            alt.Tooltip("Descrição:N", title="Descrição"),
                            alt.Tooltip("Modo operação:N", title="Modo"),
                            alt.Tooltip("Hectare:Q", title="Hectare", format=".2f"),
                            alt.Tooltip("R(t):Q", title="R(t)", format=".6f"),
                            alt.Tooltip("Eta:Q", title="Eta", format=".4f"),
                            alt.Tooltip("Beta:Q", title="Beta", format=".4f"),
                        ]
                    )
                    .properties(height=420)
                    .interactive()
                )

                st.altair_chart(graf, use_container_width=True)

                # ---------
                # Prévia da base exportável
                # ---------
                st.markdown("### Prévia da base exportável")
                st.dataframe(
                    df_export.head(50),
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Modelo": st.column_config.TextColumn("Modelo"),
                        "Modo operação": st.column_config.TextColumn("Modo operação"),
                        "Código": st.column_config.TextColumn("Código"),
                        "R(t)": st.column_config.NumberColumn("R(t)", format="%.6f"),
                        "Hectare": st.column_config.NumberColumn("Hectare", format="%.2f"),
                        "Eta": st.column_config.NumberColumn("Eta", format="%.4f"),
                        "Beta": st.column_config.NumberColumn("Beta", format="%.4f"),
                    }
                )

                buffer_conf = gerar_excel_confiabilidade(df_export)

                st.download_button(
                    label="⬇️ Exportar base de confiabilidade (.xlsx)",
                    data=buffer_conf,
                    file_name="base_confiabilidade_weibull.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )