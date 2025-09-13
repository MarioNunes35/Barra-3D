
# -*- coding: utf-8 -*-
"""
Streamlit — 3D Bar Chart (Advanced)
Author: ChatGPT
Date: 2025-09-13

Recursos principais:
- Upload de CSV/Excel e editor de dados integrado
- Mapeamento de colunas (X, Y, Z, erros)
- Barras 3D com Matplotlib (sem dependência de Chrome/Kaleido)
- Barras com coloração fixa, por série (Y) ou por altura (colormap)
- Barras com largura/profundidade/espessura da borda ajustáveis
- Rótulos de valores no topo das barras (formatação customizável)
- Erros simétricos ou assimétricos renderizados como “hastes” 3D com “chapéus”
- Controle de câmera (elevação/azimute), tema claro/escuro, grade, fundo, limites de eixos
- Exportação em PNG/SVG com DPI customizável

Requisitos:
- streamlit
- pandas
- numpy
- matplotlib
"""

import io
import math
import sys
import numpy as np
import pandas as pd
import streamlit as st
from typing import List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - necessário para 3D
from matplotlib.colors import Normalize
from matplotlib import cm


# -----------------------------
# Utilidades
# -----------------------------
def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Garante que o DataFrame possua pelo menos 3 colunas; se não, cria um exemplo."""
    if df is None or df.empty or df.shape[1] < 3:
        df = sample_dataframe()
    return df


def sample_dataframe(n_x=4, n_y=4, seed=42) -> pd.DataFrame:
    """Cria um conjunto de dados de exemplo (formato longo): X (cat), Y (cat), Z (valor), Err (simétrico)."""
    rng = np.random.default_rng(seed)
    xs = [f"Grupo {i+1}" for i in range(n_x)]
    ys = [f"Série {j+1}" for j in range(n_y)]
    data = []
    for x in xs:
        for y in ys:
            z = float(rng.uniform(10, 100))
            err = float(z * rng.uniform(0.05, 0.15))
            data.append([x, y, z, err])
    return pd.DataFrame(data, columns=["X", "Y", "Z", "ERR"])  # ERR simétrico por padrão


def detect_categorical_order(series: pd.Series) -> List:
    """Mantém a ordem de aparição, preservando duplicatas únicas na sequência."""
    seen = set()
    order = []
    for item in series.tolist():
        if item not in seen:
            seen.add(item)
            order.append(item)
    return order


def parse_order_input(text: str, detected: List) -> List:
    """Tenta parsear uma lista separada por vírgulas; se vazio, retorna detectado."""
    if not text or not text.strip():
        return detected
    parts = [p.strip() for p in text.split(",")]
    return [p for p in parts if p in set(detected)] or detected


# -----------------------------
# Renderização 3D
# -----------------------------
def draw_3d_bars(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    err_style: str = "Nenhum",  # "Nenhum" | "Simétrico" | "Assimétrico"
    err_col: Optional[str] = None,
    err_low_col: Optional[str] = None,
    err_high_col: Optional[str] = None,
    x_order: Optional[List] = None,
    y_order: Optional[List] = None,
    figsize: Tuple[float, float] = (10, 7.5),
    elev: float = 20.0,
    azim: float = -60.0,
    bar_width: float = 0.6,
    bar_depth: float = 0.6,
    alpha: float = 0.95,
    edge_on: bool = True,
    edge_width: float = 0.5,
    edge_color: str = "black",
    color_mode: str = "Única",   # "Única" | "Por série (Y)" | "Por altura (colormap)"
    base_color: str = "#3182bd",
    colormap_name: str = "viridis",
    series_palette_name: str = "tab10",
    show_values: bool = True,
    value_fmt: str = "{:.2f}",
    value_offset: float = 0.02,
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
    show_grid: bool = True,
    bg_color: str = "white",
    pane_color: str = "#f5f5f5",
    title: str = "Gráfico de Barras 3D",
    xlabel: str = "X",
    ylabel: str = "Y",
    zlabel: str = "Z",
    label_size: int = 12,
    tick_size: int = 10,
    title_size: int = 16,
    cap_ratio: float = 0.6,  # largura do “chapéu” de erro em relação à largura da barra
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Renderiza as barras 3D usando Matplotlib e retorna (fig, ax).
    """

    # Preparação dos dados e ordens
    if x_order is None:
        x_order = detect_categorical_order(df[x_col])
    if y_order is None:
        y_order = detect_categorical_order(df[y_col])

    x_index = {val: i for i, val in enumerate(x_order)}
    y_index = {val: i for i, val in enumerate(y_order)}

    # Mapeia cores de série (Y) se necessário
    if color_mode == "Por série (Y)":
        palette = plt.get_cmap(series_palette_name)
        colors_by_y = {y: palette(i / max(1, len(y_order) - 1)) for i, y in enumerate(y_order)}
    elif color_mode == "Por altura (colormap)":
        cmap = cm.get_cmap(colormap_name)
        z_vals = df[z_col].astype(float).values
        z_min = float(np.min(z_vals)) if zmin is None else zmin
        z_max = float(np.max(z_vals)) if zmax is None else zmax
        norm = Normalize(vmin=z_min, vmax=z_max)

    # Figura/Axes
    fig = plt.figure(figsize=figsize, facecolor=bg_color)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)

    # Ajustes visuais
    ax.w_xaxis.set_pane_color(matplotlib.colors.to_rgba(pane_color, 1.0))
    ax.w_yaxis.set_pane_color(matplotlib.colors.to_rgba(pane_color, 1.0))
    ax.w_zaxis.set_pane_color(matplotlib.colors.to_rgba(pane_color, 1.0))
    ax.grid(show_grid)

    # Construção dos vetores para bar3d
    xs = []
    ys = []
    zs = []  # base (normalmente 0)
    dx = []
    dy = []
    dz = []
    facecolors = []

    # Para rótulos de valores e erros
    value_labels = []
    error_segments = []  # lista de tuplas: (x, y, z_bottom, z_top, cap_half)

    for _, row in df.iterrows():
        xv = x_index[row[x_col]]
        yv = y_index[row[y_col]]
        z = float(row[z_col])
        xs.append(xv - bar_width / 2.0)
        ys.append(yv - bar_depth / 2.0)
        zs.append(0.0)  # base
        dx.append(bar_width)
        dy.append(bar_depth)
        dz.append(z)

        # Determina a cor
        if color_mode == "Única":
            facecolors.append(base_color)
        elif color_mode == "Por série (Y)":
            facecolors.append(colors_by_y[row[y_col]])
        else:  # "Por altura (colormap)"
            facecolors.append(cmap(norm(z)))

        # Guarda info para rótulo e possíveis erros
        x_center = xv
        y_center = yv
        value_labels.append((x_center, y_center, z))

        if err_style == "Simétrico" and err_col and not pd.isna(row.get(err_col, np.nan)):
            e = float(row[err_col])
            error_segments.append((x_center, y_center, z - e, z + e))
        elif err_style == "Assimétrico" and (err_low_col and err_high_col):
            low = row.get(err_low_col, np.nan)
            high = row.get(err_high_col, np.nan)
            if not (pd.isna(low) or pd.isna(high)):
                low = float(low)
                high = float(high)
                error_segments.append((x_center, y_center, z - low, z + high))

    # Desenha barras
    ax.bar3d(xs, ys, zs, dx, dy, dz, shade=True, color=facecolors, edgecolor=edge_color if edge_on else None, linewidth=edge_width, alpha=alpha)

    # Ticks e rótulos de eixos
    ax.set_title(title, fontsize=title_size, pad=12)
    ax.set_xlabel(xlabel, fontsize=label_size, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=label_size, labelpad=10)
    ax.set_zlabel(zlabel, fontsize=label_size, labelpad=10)

    ax.set_xticks(range(len(x_order)))
    ax.set_xticklabels(x_order, fontsize=tick_size, rotation=0, ha="center")
    ax.set_yticks(range(len(y_order)))
    ax.set_yticklabels(y_order, fontsize=tick_size, rotation=0, ha="center")

    # Limites Z
    if zmin is not None or zmax is not None:
        current = ax.get_zlim()
        zlo = zmin if zmin is not None else current[0]
        zhi = zmax if zmax is not None else current[1]
        if zhi <= zlo:
            zhi = zlo + 1.0
        ax.set_zlim(zlo, zhi)

    # Rótulos de valores no topo das barras
    if show_values:
        for (xc, yc, z) in value_labels:
            ax.text(
                xc,
                yc,
                z + max(0.001, value_offset * (ax.get_zlim()[1] - ax.get_zlim()[0])),
                value_fmt.format(z),
                ha="center",
                va="bottom",
                fontsize=tick_size,
                color="black",
                zdir=None,
            )

    # Erros: hastes + chapéus
    if error_segments:
        cap_half = (bar_width * cap_ratio) / 2.0
        for (xc, yc, z_bottom, z_top) in error_segments:
            # haste vertical
            ax.plot([xc, xc], [yc, yc], [z_bottom, z_top], color=edge_color, linewidth=edge_width + 0.2, alpha=min(1.0, alpha + 0.05))
            # chapéus
            ax.plot([xc - cap_half, xc + cap_half], [yc, yc], [z_top, z_top], color=edge_color, linewidth=edge_width + 0.2, alpha=min(1.0, alpha + 0.05))
            ax.plot([xc - cap_half, xc + cap_half], [yc, yc], [z_bottom, z_bottom], color=edge_color, linewidth=edge_width + 0.2, alpha=min(1.0, alpha + 0.05))

    # Fundo
    fig.patch.set_facecolor(bg_color)

    return fig, ax


def export_figure(fig: plt.Figure, fmt: str = "png", dpi: int = 300) -> bytes:
    bio = io.BytesIO()
    fig.savefig(bio, format=fmt, dpi=dpi, bbox_inches="tight")
    bio.seek(0)
    return bio.read()


# -----------------------------
# App Streamlit
# -----------------------------
def main():
    st.set_page_config(page_title="Barras 3D — Avançado", layout="wide")
    st.title("Gráfico de Barras 3D — Opções Avançadas")

    with st.expander("📥 Dados", expanded=True):
        col_u1, col_u2 = st.columns([1, 1])
        with col_u1:
            up = st.file_uploader("Envie um CSV ou Excel (long format, com colunas para X, Y, Z e erros opcionais)", type=["csv", "xlsx"])
            sep = st.text_input("Separador (apenas para CSV)", value=",")
        with col_u2:
            st.caption("Dica: se não enviar nenhum arquivo, um dataset de exemplo será carregado.")
            use_editor = st.checkbox("Editar dados com data editor (recomendado para pequenos ajustes)", value=False)

        df = None
        if up is not None:
            try:
                if up.name.lower().endswith(".csv"):
                    df = pd.read_csv(up, sep=sep)
                else:
                    df = pd.read_excel(up)
            except Exception as e:
                st.error(f"Falha ao ler arquivo: {e}")
                df = None

        df = ensure_columns(df)
        st.write("Prévia do DataFrame original:")
        st.dataframe(df, use_container_width=True)

        if use_editor:
            df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

        # Mapeamento de colunas
        cols = list(df.columns)
        st.subheader("Mapeamento de Colunas")
        c1, c2, c3 = st.columns(3)
        with c1:
            x_col = st.selectbox("Coluna X (categoria ou número)", options=cols, index=0)
        with c2:
            y_col = st.selectbox("Coluna Y (categoria ou número)", options=cols, index=1 if len(cols) > 1 else 0)
        with c3:
            z_col = st.selectbox("Coluna Z (altura)", options=cols, index=2 if len(cols) > 2 else 0)

        st.markdown("---")
        st.subheader("Erros (opcional)")
        err_style = st.radio("Tipo de erro", options=["Nenhum", "Simétrico", "Assimétrico"], horizontal=True, index=0)
        err_col = err_low_col = err_high_col = None
        if err_style == "Simétrico":
            err_col = st.selectbox("Coluna de erro (±)", options=["(nenhuma)"] + cols, index=0)
            if err_col == "(nenhuma)":
                err_col = None
        elif err_style == "Assimétrico":
            c1e, c2e = st.columns(2)
            with c1e:
                err_low_col = st.selectbox("Erro inferior", options=["(nenhuma)"] + cols, index=0)
                if err_low_col == "(nenhuma)":
                    err_low_col = None
            with c2e:
                err_high_col = st.selectbox("Erro superior", options=["(nenhuma)"] + cols, index=0)
                if err_high_col == "(nenhuma)":
                    err_high_col = None

        st.markdown("---")
        st.subheader("Ordenação (categorias)")
        detected_x = detect_categorical_order(df[x_col])
        detected_y = detect_categorical_order(df[y_col])
        ox = st.text_input("Ordem do eixo X (separada por vírgulas). Deixe vazio para detectar automaticamente.", value="")
        oy = st.text_input("Ordem do eixo Y (separada por vírgulas). Deixe vazio para detectar automaticamente.", value="")
        x_order = parse_order_input(ox, detected_x)
        y_order = parse_order_input(oy, detected_y)

    with st.expander("🎨 Personalização do Gráfico", expanded=True):
        st.subheader("Estilo e Câmera")
        cs1, cs2, cs3, cs4 = st.columns(4)
        with cs1:
            elev = st.slider("Elevação da câmera (elev)", 0, 90, 20)
        with cs2:
            azim = st.slider("Azimute da câmera (azim)", -180, 180, -60)
        with cs3:
            width_in = st.number_input("Largura da figura (pol.)", 4.0, 24.0, 10.0, step=0.5)
        with cs4:
            height_in = st.number_input("Altura da figura (pol.)", 3.0, 20.0, 7.5, step=0.5)

        st.subheader("Barras")
        cb1, cb2, cb3, cb4 = st.columns(4)
        with cb1:
            bar_width = st.slider("Largura da barra (dx)", 0.1, 1.0, 0.6, step=0.05)
        with cb2:
            bar_depth = st.slider("Profundidade da barra (dy)", 0.1, 1.0, 0.6, step=0.05)
        with cb3:
            alpha = st.slider("Opacidade (alpha)", 0.1, 1.0, 0.95, step=0.05)
        with cb4:
            edge_on = st.checkbox("Borda ativada", value=True)
        cb5, cb6, cb7 = st.columns([1,1,2])
        with cb5:
            edge_width = st.number_input("Espessura da borda", 0.0, 5.0, 0.5, step=0.1)
        with cb6:
            edge_color = st.color_picker("Cor da borda", value="#000000")
        with cb7:
            color_mode = st.selectbox("Modo de cor", ["Única", "Por série (Y)", "Por altura (colormap)"], index=0)

        cc1, cc2, cc3 = st.columns(3)
        base_color = "#3182bd"
        colormap_name = "viridis"
        series_palette_name = "tab10"
        with cc1:
            if color_mode == "Única":
                base_color = st.color_picker("Cor base", value=base_color)
            elif color_mode == "Por altura (colormap)":
                colormap_name = st.selectbox("Colormap", sorted(m for m in plt.colormaps() if not m.endswith("_r")), index=sorted(m for m in plt.colormaps() if not m.endswith("_r")).index("viridis"))
        with cc2:
            if color_mode == "Por série (Y)":
                series_palette_name = st.selectbox("Paleta de séries (Y)", options=["tab10", "tab20", "Set3", "Pastel1", "Accent"], index=0)
        with cc3:
            st.caption("Dica: escolha 'Por altura (colormap)' para mapear cores pela magnitude do Z.")

        st.subheader("Rótulos e Legendas")
        rl1, rl2, rl3 = st.columns(3)
        with rl1:
            title = st.text_input("Título", value="Gráfico de Barras 3D")
            xlabel = st.text_input("Rótulo eixo X", value="X")
        with rl2:
            ylabel = st.text_input("Rótulo eixo Y", value="Y")
            zlabel = st.text_input("Rótulo eixo Z", value="Z")
        with rl3:
            title_size = st.slider("Tamanho do título", 8, 40, 16)
            label_size = st.slider("Tamanho rótulos de eixo", 6, 30, 12)
            tick_size = st.slider("Tamanho dos ticks", 6, 30, 10)

        st.subheader("Valores e Erros")
        ve1, ve2, ve3, ve4 = st.columns(4)
        with ve1:
            show_values = st.checkbox("Mostrar valores no topo", value=True)
        with ve2:
            value_fmt = st.text_input("Formato do valor", value="{:.2f}")
        with ve3:
            value_offset = st.number_input("Offset vertical do valor (fração do eixo Z)", 0.0, 0.2, 0.02, step=0.005)
        with ve4:
            cap_ratio = st.slider("Largura do chapéu do erro (relativa à barra)", 0.2, 1.2, 0.6, step=0.05)

        st.subheader("Fundo e Eixos")
        fe1, fe2, fe3 = st.columns(3)
        with fe1:
            bg_color = st.color_picker("Cor de fundo", value="#ffffff")
        with fe2:
            pane_color = st.color_picker("Cor do plano (pane)", value="#f5f5f5")
        with fe3:
            show_grid = st.checkbox("Mostrar grade", value=True)

        st.subheader("Limites de Z (opcional)")
        lz1, lz2 = st.columns(2)
        with lz1:
            zmin_en = st.checkbox("Definir Zmín", value=False)
            zmin_val = st.number_input("Z mín (se ativado)", value=0.0, step=1.0)
        with lz2:
            zmax_en = st.checkbox("Definir Zmáx", value=False)
            zmax_val = st.number_input("Z máx (se ativado)", value=100.0, step=1.0)

    # Render
    st.markdown("---")
    st.subheader("📊 Resultado")
    zmin = zmin_val if zmin_en else None
    zmax = zmax_val if zmax_en else None

    fig, ax = draw_3d_bars(
        df=df,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        err_style=err_style,
        err_col=err_col,
        err_low_col=err_low_col,
        err_high_col=err_high_col,
        x_order=x_order,
        y_order=y_order,
        figsize=(width_in, height_in),
        elev=elev,
        azim=azim,
        bar_width=bar_width,
        bar_depth=bar_depth,
        alpha=alpha,
        edge_on=edge_on,
        edge_width=edge_width,
        edge_color=edge_color,
        color_mode=color_mode,
        base_color=base_color,
        colormap_name=colormap_name,
        series_palette_name=series_palette_name,
        show_values=show_values,
        value_fmt=value_fmt,
        value_offset=value_offset,
        zmin=zmin,
        zmax=zmax,
        show_grid=show_grid,
        bg_color=bg_color,
        pane_color=pane_color,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        zlabel=zlabel,
        label_size=label_size,
        tick_size=tick_size,
        title_size=title_size,
        cap_ratio=cap_ratio,
    )

    st.pyplot(fig, use_container_width=True)

    with st.expander("💾 Exportar figura"):
        ex1, ex2, ex3 = st.columns(3)
        with ex1:
            fmt = st.selectbox("Formato", options=["png", "svg"], index=0)
        with ex2:
            dpi = st.slider("DPI (apenas PNG)", 72, 600, 300, step=10)
        with ex3:
            fname = st.text_input("Nome do arquivo (sem extensão)", value="barras3d_advanced")

        bio = io.BytesIO()
        if fmt == "png":
            fig.savefig(bio, format="png", dpi=dpi, bbox_inches="tight")
            mime = "image/png"
            ext = "png"
        else:
            fig.savefig(bio, format="svg", bbox_inches="tight")
            mime = "image/svg+xml"
            ext = "svg"
        bio.seek(0)
        st.download_button(
            label=f"⬇️ Baixar {ext.upper()}",
            data=bio,
            file_name=f"{fname}.{ext}",
            mime=mime,
            use_container_width=True
        )

    st.markdown("---")
    st.caption("Dica: para usar no Streamlit Cloud, adicione `streamlit`, `pandas`, `numpy` e `matplotlib` ao requirements.txt.")

if __name__ == "__main__":
    main()



