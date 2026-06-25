import io
import re
import html
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Survey Analysis Workspace",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "Survey Analysis Workspace"
APP_SUBTITLE = "Загрузите Excel/CSV с ответами на опрос и получите topline, crosstabs, heatmaps, корреляции и анализ открытых ответов."

QUESTION_TYPES = ["single_choice", "multiple_choice", "rating", "open_text", "other_text", "numeric", "ignore"]
METRICS_CAT = ["Количество", "% по колонке", "% по строке", "% от всех"]
METRICS_RATING = ["Среднее", "Медиана", "Top-2-box", "Bottom-2-box", "N"]
BLOCK_NAMES = {
    "1": "Профиль",
    "2": "Макросы и заготовки",
    "3": "Переменные",
    "4": "Обучение и поддержка",
    "5": "Процесс разработки",
    "6": "Техническое состояние",
    "7": "Визуализация",
    "8": "Конфигуратор систем",
    "9": "Командная работа",
    "10": "Общие комментарии",
}


def strip_html_tags(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_option(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    text = strip_html_tags(value).strip()
    text = re.sub(r"\s+", " ", text)
    if text == "":
        return None
    return text


def extract_question_number(text: str) -> str:
    match = re.search(r"\b(\d{1,2}\.\d{1,2})\b", text)
    return match.group(1) if match else ""


def question_block(question_number: str) -> str:
    if not question_number:
        return "Без блока"
    top = question_number.split(".")[0]
    return BLOCK_NAMES.get(top, f"Блок {top}")


def simplify_question_text(text: str) -> str:
    text = strip_html_tags(text)
    # Убираем длинные пояснения шкал, чтобы в интерфейсе были читаемые заголовки.
    text = re.sub(r"\s*\(1\s*[–-].*$", "", text).strip()
    text = re.sub(r"\s*\(где\s+1\s*[–-].*$", "", text, flags=re.IGNORECASE).strip()
    return text




def shorten_label(value: object, max_chars: int = 34, prefer_question_number: bool = False) -> str:
    """Короткая подпись для осей графиков. Полный текст остаётся в hover/таблице."""
    text = simplify_question_text(value)
    if not text:
        return ""
    number = extract_question_number(text)
    if prefer_question_number and number:
        return number
    if number and len(text) > max_chars:
        # Для вопросов оставляем номер: он нужен как якорь для расшифровки под графиком.
        tail = re.sub(r"^\s*" + re.escape(number) + r"\s*[.\-–—:]*\s*", "", text).strip()
        available = max(8, max_chars - len(number) - 2)
        if len(tail) > available:
            tail = tail[: available - 1].rstrip() + "…"
        return f"{number}. {tail}"
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def make_unique_labels(values: List[object], max_chars: int = 34, prefer_question_number: bool = False) -> List[str]:
    labels: List[str] = []
    seen: Dict[str, int] = {}
    for value in values:
        label = shorten_label(value, max_chars=max_chars, prefer_question_number=prefer_question_number)
        if not label:
            label = "—"
        if label in seen:
            seen[label] += 1
            label = f"{label} [{seen[label]}]"
        else:
            seen[label] = 1
        labels.append(label)
    return labels


def format_heatmap_text(matrix: pd.DataFrame) -> List[List[str]]:
    result: List[List[str]] = []
    for row in matrix.to_numpy(dtype=float):
        formatted_row: List[str] = []
        for value in row:
            if pd.isna(value):
                formatted_row.append("")
            elif abs(value) >= 100:
                formatted_row.append(f"{value:.0f}")
            elif abs(value) >= 10:
                formatted_row.append(f"{value:.1f}")
            else:
                formatted_row.append(f"{value:.2f}".rstrip("0").rstrip("."))
        result.append(formatted_row)
    return result


def readable_heatmap(
    matrix: pd.DataFrame,
    title: str,
    value_name: str = "Значение",
    row_kind: str = "Вопрос",
    col_kind: str = "Сегмент",
    prefer_question_numbers_on_y: bool = True,
    max_x_label_chars: int = 22,
    max_y_label_chars: int = 38,
    show_values: bool = True,
    height: Optional[int] = None,
) -> go.Figure:
    """Plotly heatmap с короткими осями и полными подписями в наведении."""
    x_full = [str(x) for x in matrix.columns.tolist()]
    y_full = [str(y) for y in matrix.index.tolist()]
    x_short = make_unique_labels(x_full, max_chars=max_x_label_chars, prefer_question_number=False)
    y_short = make_unique_labels(y_full, max_chars=max_y_label_chars, prefer_question_number=prefer_question_numbers_on_y)

    z = matrix.astype(float).to_numpy()
    customdata = []
    for y in y_full:
        row = []
        for x in x_full:
            row.append([y, x])
        customdata.append(row)

    text = format_heatmap_text(matrix) if show_values else None
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x_short,
            y=y_short,
            customdata=customdata,
            text=text,
            texttemplate="%{text}" if show_values else None,
            textfont={"size": 11},
            colorscale="Blues",
            colorbar={"title": value_name},
            hovertemplate=(
                f"<b>{row_kind}</b>: %{{customdata[0]}}<br>"
                f"<b>{col_kind}</b>: %{{customdata[1]}}<br>"
                f"<b>{value_name}</b>: %{{z:.3g}}<extra></extra>"
            ),
        )
    )
    if height is None:
        height = max(460, min(1000, 220 + 38 * max(1, matrix.shape[0])))
    tickangle = -35 if any(len(str(x)) > 12 for x in x_short) else 0
    fig.update_layout(
        title=title,
        height=height,
        margin={"l": 110, "r": 40, "t": 80, "b": 120},
        xaxis={"tickangle": tickangle, "automargin": True},
        yaxis={"automargin": True},
    )
    return fig


def label_mapping_df(values: List[object], label_type: str, max_chars: int = 38, prefer_question_number: bool = False) -> pd.DataFrame:
    full = [str(v) for v in values]
    short = make_unique_labels(full, max_chars=max_chars, prefer_question_number=prefer_question_number)
    return pd.DataFrame({"Тип": label_type, "Подпись на графике": short, "Полный текст": full})


def make_unique_columns(cols: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    result: List[str] = []
    for col in cols:
        base = simplify_question_text(col)
        if not base:
            base = "Без названия"
        if base not in seen:
            seen[base] = 1
            result.append(base)
        else:
            seen[base] += 1
            result.append(f"{base} [{seen[base]}]")
    return result


def split_multi(value: object) -> List[str]:
    if pd.isna(value):
        return []
    text = strip_html_tags(value)
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"\s*;\s*", text) if p.strip()]
    return parts


def is_probably_junk_text(text: str) -> bool:
    if text is None:
        return True
    t = str(text).strip().lower()
    if t in {"", "-", "--", "нет", "нету", "не", "н/д", "na", "n/a", "test", "тест"}:
        return True
    if re.fullmatch(r"[\d\W_]+", t) and len(t) <= 8:
        return True
    if len(t) <= 1:
        return True
    return False


def infer_question_type(series: pd.Series, col_name: str) -> str:
    col_clean = simplify_question_text(col_name).lower()
    values = series.dropna().map(clean_option).dropna()
    values = values[values != ""]
    n = len(values)
    if n == 0:
        return "ignore"

    if "другое" in col_clean or "другие" in col_clean:
        return "other_text"

    semicolon_share = values.astype(str).str.contains(";", regex=False).mean()
    if semicolon_share >= 0.12:
        return "multiple_choice"

    numeric = pd.to_numeric(values.astype(str).str.replace(",", ".", regex=False), errors="coerce")
    numeric_share = numeric.notna().mean()
    unique_numeric = sorted(numeric.dropna().unique().tolist())
    if numeric_share >= 0.75:
        if len(unique_numeric) <= 7 and min(unique_numeric) >= 0 and max(unique_numeric) <= 10:
            # Большинство шкал в таких опросах 1-5 или 0-10.
            return "rating"
        return "numeric"

    unique_count = values.nunique(dropna=True)
    unique_share = unique_count / max(n, 1)
    avg_len = values.astype(str).map(len).mean()

    if unique_count <= 18 and unique_share <= 0.35:
        return "single_choice"
    if avg_len >= 35 or unique_share >= 0.45:
        return "open_text"
    return "single_choice"


def build_question_catalog(df: pd.DataFrame, original_cols: List[str]) -> pd.DataFrame:
    rows = []
    cleaned_cols = list(df.columns)
    for i, (orig, clean) in enumerate(zip(original_cols, cleaned_cols), start=1):
        qnum = extract_question_number(clean)
        qtype = infer_question_type(df[clean], orig)
        block = question_block(qnum)
        is_segment = block == "Профиль" and qtype in ["single_choice", "multiple_choice"]
        is_outcome = qtype == "rating" and block in ["Макросы и заготовки", "Техническое состояние", "Конфигуратор систем"]
        rows.append(
            {
                "question_id": f"Q{i:02d}",
                "number": qnum,
                "column": clean,
                "original_column": strip_html_tags(orig),
                "text": clean,
                "type": qtype,
                "block": block,
                "is_segment": is_segment,
                "is_outcome": is_outcome,
                "include": qtype != "ignore",
            }
        )
    return pd.DataFrame(rows)


def read_uploaded_file(uploaded_file, sheet_name: Optional[str] = None) -> Tuple[pd.DataFrame, List[str], List[str]]:
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        # Пробуем две частые кодировки, потому что CSV, видимо, создан для тренировки терпения.
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python")
        except Exception:
            df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python", encoding="cp1251")
        sheet_names = ["CSV"]
    else:
        xls = pd.ExcelFile(io.BytesIO(file_bytes))
        sheet_names = xls.sheet_names
        sheet = sheet_name or sheet_names[0]
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet)
    original_cols = [str(c) for c in df.columns]
    df.columns = make_unique_columns(original_cols)
    return df, original_cols, sheet_names


def get_catalog() -> pd.DataFrame:
    return st.session_state.get("catalog", pd.DataFrame())


def get_df() -> pd.DataFrame:
    return st.session_state.get("df", pd.DataFrame())


def question_by_type(types: List[str], include_segments: bool = False) -> List[str]:
    catalog = get_catalog()
    if catalog.empty:
        return []
    mask = catalog["include"] & catalog["type"].isin(types)
    if include_segments:
        mask = mask | (catalog["include"] & catalog["is_segment"])
    return catalog.loc[mask, "column"].tolist()


def question_type(column: str) -> str:
    catalog = get_catalog()
    if catalog.empty or column not in catalog["column"].values:
        return "ignore"
    return catalog.loc[catalog["column"] == column, "type"].iloc[0]


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def top2_bottom2(values: pd.Series) -> Tuple[float, float]:
    nums = numeric_series(values).dropna()
    if nums.empty:
        return np.nan, np.nan
    max_val = nums.max()
    min_val = nums.min()
    if max_val <= 5:
        top2 = nums.isin([4, 5]).mean() * 100
        bottom2 = nums.isin([1, 2]).mean() * 100
    elif max_val <= 10:
        top2 = (nums >= 9).mean() * 100
        bottom2 = (nums <= 6).mean() * 100
    else:
        q75 = nums.quantile(0.75)
        q25 = nums.quantile(0.25)
        top2 = (nums >= q75).mean() * 100
        bottom2 = (nums <= q25).mean() * 100
    return float(top2), float(bottom2)


def rating_summary(series: pd.Series) -> Dict[str, object]:
    nums = numeric_series(series).dropna()
    top2, bottom2 = top2_bottom2(series)
    return {
        "N": int(nums.count()),
        "Среднее": round(float(nums.mean()), 2) if nums.count() else np.nan,
        "Медиана": round(float(nums.median()), 2) if nums.count() else np.nan,
        "Ст. отклонение": round(float(nums.std(ddof=1)), 2) if nums.count() > 1 else np.nan,
        "Top-2-box, %": round(top2, 1) if not pd.isna(top2) else np.nan,
        "Bottom-2-box, %": round(bottom2, 1) if not pd.isna(bottom2) else np.nan,
        "Пропуски": int(series.isna().sum()),
    }


def categorical_counts(series: pd.Series, multi: bool = False) -> pd.DataFrame:
    if multi:
        exploded = series.dropna().apply(split_multi).explode().dropna()
        counts = exploded.value_counts().reset_index()
        counts.columns = ["Вариант", "Количество"]
        base_n = series.dropna().shape[0]
        counts["% респондентов"] = counts["Количество"] / max(base_n, 1) * 100
        counts["% респондентов"] = counts["% респондентов"].round(1)
        return counts
    values = series.dropna().map(clean_option).dropna()
    counts = values.value_counts().reset_index()
    counts.columns = ["Вариант", "Количество"]
    counts["%"] = (counts["Количество"] / max(values.shape[0], 1) * 100).round(1)
    return counts


def explode_question(df: pd.DataFrame, column: str) -> pd.DataFrame:
    qtype = question_type(column)
    tmp = df[[column]].copy()
    tmp["__row_id"] = np.arange(len(tmp))
    if qtype == "multiple_choice":
        tmp[column] = tmp[column].apply(split_multi)
        tmp = tmp.explode(column)
        tmp[column] = tmp[column].map(clean_option)
        tmp = tmp.dropna(subset=[column])
    else:
        tmp[column] = tmp[column].map(clean_option)
        tmp = tmp.dropna(subset=[column])
    return tmp[["__row_id", column]]


def crosstab_categorical(df: pd.DataFrame, row_q: str, col_q: str, metric: str) -> pd.DataFrame:
    left = explode_question(df, row_q).rename(columns={row_q: "__row_value"})
    right = explode_question(df, col_q).rename(columns={col_q: "__col_value"})
    merged = left.merge(right, on="__row_id", how="inner")
    if merged.empty:
        return pd.DataFrame()
    table = pd.crosstab(merged["__row_value"], merged["__col_value"])
    if metric == "% по колонке":
        table = table.div(table.sum(axis=0).replace(0, np.nan), axis=1) * 100
    elif metric == "% по строке":
        table = table.div(table.sum(axis=1).replace(0, np.nan), axis=0) * 100
    elif metric == "% от всех":
        table = table / max(table.to_numpy().sum(), 1) * 100
    if metric != "Количество":
        table = table.round(1)
    table.index.name = row_q
    table.columns.name = col_q
    return table


def rating_by_segment(df: pd.DataFrame, rating_q: str, segment_q: str, metric: str) -> pd.DataFrame:
    seg = explode_question(df, segment_q)
    vals = df[[rating_q]].copy()
    vals["__row_id"] = np.arange(len(vals))
    vals["__value"] = numeric_series(vals[rating_q])
    merged = seg.merge(vals[["__row_id", "__value"]], on="__row_id", how="inner").dropna(subset=["__value"])
    if merged.empty:
        return pd.DataFrame()
    if metric == "Среднее":
        out = merged.groupby(segment_q)["__value"].mean().round(2)
    elif metric == "Медиана":
        out = merged.groupby(segment_q)["__value"].median().round(2)
    elif metric == "Top-2-box":
        max_val = merged["__value"].max()
        if max_val <= 5:
            out = merged.assign(__top=merged["__value"].isin([4, 5])).groupby(segment_q)["__top"].mean() * 100
        else:
            out = merged.assign(__top=merged["__value"] >= 9).groupby(segment_q)["__top"].mean() * 100
        out = out.round(1)
    elif metric == "Bottom-2-box":
        max_val = merged["__value"].max()
        if max_val <= 5:
            out = merged.assign(__bottom=merged["__value"].isin([1, 2])).groupby(segment_q)["__bottom"].mean() * 100
        else:
            out = merged.assign(__bottom=merged["__value"] <= 6).groupby(segment_q)["__bottom"].mean() * 100
        out = out.round(1)
    else:
        out = merged.groupby(segment_q)["__value"].count()
    return out.to_frame(name=metric).sort_values(metric, ascending=False)


def build_topline_table(df: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, q in catalog[catalog["include"]].iterrows():
        col = q["column"]
        qtype = q["type"]
        series = df[col]
        base = {
            "question_id": q["question_id"],
            "number": q["number"],
            "block": q["block"],
            "question": col,
            "type": qtype,
            "N valid": int(series.notna().sum()),
            "Missing": int(series.isna().sum()),
        }
        if qtype in ["rating", "numeric"]:
            base.update(rating_summary(series))
        elif qtype in ["single_choice", "multiple_choice"]:
            counts = categorical_counts(series, multi=(qtype == "multiple_choice"))
            base["Вариантов"] = int(counts.shape[0])
            if not counts.empty:
                base["Топ-вариант"] = counts.iloc[0, 0]
                base["Доля топ-варианта, %"] = counts.iloc[0, 2]
        elif qtype in ["open_text", "other_text"]:
            texts = series.dropna().astype(str).map(strip_html_tags)
            texts = texts[~texts.map(is_probably_junk_text)]
            base["Текстовых ответов"] = int(texts.shape[0])
        rows.append(base)
    return pd.DataFrame(rows)


def export_dataframes_to_excel(named_tables: Dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for name, table in named_tables.items():
            safe_name = re.sub(r"[\\/*?:\[\]]", "_", name)[:31] or "Sheet"
            table.to_excel(writer, sheet_name=safe_name, index=True)
            worksheet = writer.sheets[safe_name]
            worksheet.freeze_panes(1, 1)
            for i, col in enumerate(table.reset_index().columns):
                width = min(max(12, len(str(col)) + 2), 50)
                worksheet.set_column(i, i, width)
    return output.getvalue()


def build_open_text_word_counts(series: pd.Series, min_len: int = 4) -> pd.DataFrame:
    texts = series.dropna().astype(str).map(strip_html_tags)
    texts = texts[~texts.map(is_probably_junk_text)]
    stop = {
        "это", "как", "что", "для", "или", "при", "над", "под", "без", "уже", "ещё", "еще", "было", "есть", "нет", "очень", "нужно",
        "надо", "можно", "чтобы", "если", "так", "все", "всё", "она", "они", "его", "мне", "нам", "вам", "более", "менее",
        "owen", "logic", "овен", "логик", "проект", "проекта", "проектов", "работы", "работать",
    }
    words: List[str] = []
    for text in texts:
        for w in re.findall(r"[A-Za-zА-Яа-яЁё0-9_\-]+", text.lower()):
            if len(w) >= min_len and w not in stop:
                words.append(w)
    if not words:
        return pd.DataFrame(columns=["Слово", "Количество"])
    out = pd.Series(words).value_counts().head(100).reset_index()
    out.columns = ["Слово", "Количество"]
    return out


def render_upload_page():
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    uploaded = st.file_uploader("Загрузите файл опроса", type=["xlsx", "xls", "csv"])
    if uploaded is None:
        st.info("Поддерживаются Excel и CSV. Для формата как в вашем файле выберите лист с ответами, обычно он называется «Ответы».")
        return

    if uploaded.name.lower().endswith(".csv"):
        df, original_cols, sheet_names = read_uploaded_file(uploaded)
        selected_sheet = "CSV"
    else:
        file_bytes = uploaded.getvalue()
        xls = pd.ExcelFile(io.BytesIO(file_bytes))
        sheet_names = xls.sheet_names
        selected_sheet = st.selectbox("Лист с ответами", sheet_names, index=0 if "Ответы" not in sheet_names else sheet_names.index("Ответы"))
        uploaded.seek(0)
        df, original_cols, _ = read_uploaded_file(uploaded, sheet_name=selected_sheet)

    st.success(f"Файл прочитан: {df.shape[0]} строк, {df.shape[1]} колонок. Лист: {selected_sheet}")
    st.dataframe(df.head(10), use_container_width=True)

    if st.button("Распознать структуру опроса", type="primary"):
        catalog = build_question_catalog(df, original_cols)
        st.session_state["df"] = df
        st.session_state["original_cols"] = original_cols
        st.session_state["catalog"] = catalog
        st.session_state["uploaded_name"] = uploaded.name
        st.session_state["sheet_name"] = selected_sheet
        st.rerun()


def render_sidebar():
    st.sidebar.title("Навигация")
    has_data = "df" in st.session_state and "catalog" in st.session_state
    if has_data:
        st.sidebar.success("Данные загружены")
        st.sidebar.caption(st.session_state.get("uploaded_name", ""))
    page = st.sidebar.radio(
        "Раздел",
        [
            "1. Загрузка",
            "2. Словарь вопросов",
            "3. Качество данных",
            "4. Topline",
            "5. Crosstabs",
            "6. Heatmaps",
            "7. Корреляции и драйверы",
            "8. Открытые ответы",
            "9. Экспорт",
        ],
    )
    return page


def require_data() -> bool:
    if "df" not in st.session_state or "catalog" not in st.session_state:
        st.warning("Сначала загрузите файл и распознайте структуру опроса.")
        return False
    return True


def render_catalog_page():
    if not require_data():
        return
    st.header("Словарь вопросов")
    st.write("Проверьте автоопределение типов. Это важный шаг: если здесь ошибка, дальше программа будет уверенно строить красивые неправильные графики. Человеческая цивилизация уже настрадалась от таких графиков.")
    catalog = get_catalog().copy()
    edited = st.data_editor(
        catalog,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "type": st.column_config.SelectboxColumn("type", options=QUESTION_TYPES),
            "block": st.column_config.TextColumn("block"),
            "is_segment": st.column_config.CheckboxColumn("is_segment"),
            "is_outcome": st.column_config.CheckboxColumn("is_outcome"),
            "include": st.column_config.CheckboxColumn("include"),
            "original_column": st.column_config.TextColumn("original_column", disabled=True),
            "question_id": st.column_config.TextColumn("question_id", disabled=True),
        },
        disabled=["question_id", "number", "column", "original_column"],
        key="catalog_editor",
    )
    if st.button("Сохранить словарь вопросов", type="primary"):
        st.session_state["catalog"] = edited
        st.success("Словарь сохранён.")

    st.subheader("Сводка распознавания")
    counts = edited["type"].value_counts().reset_index()
    counts.columns = ["Тип", "Количество"]
    st.dataframe(counts, use_container_width=True, hide_index=True)


def render_quality_page():
    if not require_data():
        return
    st.header("Качество данных")
    df = get_df()
    catalog = get_catalog()
    rating_cols = catalog.loc[catalog["include"] & catalog["type"].eq("rating"), "column"].tolist()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ответов", f"{df.shape[0]}")
    col2.metric("Вопросов", f"{catalog[catalog['include']].shape[0]}")
    col3.metric("Полных дублей", f"{int(df.duplicated().sum())}")
    missing_share = df.isna().mean(axis=1)
    col4.metric("Анкет с >50% пропусков", f"{int((missing_share > 0.5).sum())}")

    if rating_cols:
        rating_num = df[rating_cols].apply(numeric_series)
        # Straightlining: 5+ шкальных ответов и почти все одинаковые.
        rating_counts = rating_num.notna().sum(axis=1)
        rating_std = rating_num.std(axis=1, skipna=True)
        straight = (rating_counts >= 5) & (rating_std.fillna(999) == 0)
        st.metric("Подозрение на straightlining по шкалам", f"{int(straight.sum())}")
    else:
        straight = pd.Series(False, index=df.index)

    quality = pd.DataFrame(
        {
            "row_number": np.arange(2, df.shape[0] + 2),
            "missing_share": missing_share.round(2),
            "is_duplicate": df.duplicated(),
            "possible_straightlining": straight,
        }
    )
    st.dataframe(quality.sort_values(["is_duplicate", "possible_straightlining", "missing_share"], ascending=False), use_container_width=True, hide_index=True)


def render_topline_page():
    if not require_data():
        return
    st.header("Topline: автоматическая сводка по всем вопросам")
    df = get_df()
    catalog = get_catalog()
    topline = build_topline_table(df, catalog)
    st.session_state["topline"] = topline

    filters = st.multiselect("Фильтр по блокам", sorted(catalog["block"].unique().tolist()), default=[])
    display = topline.copy()
    if filters:
        display = display[display["block"].isin(filters)]
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.subheader("Карточка вопроса")
    q_options = catalog.loc[catalog["include"], "column"].tolist()
    if not q_options:
        st.info("Нет включённых вопросов.")
        return
    q = st.selectbox("Выберите вопрос", q_options)
    qtype = question_type(q)
    st.caption(f"Тип: {qtype}")

    if qtype in ["rating", "numeric"]:
        summary = rating_summary(df[q])
        cols = st.columns(6)
        for i, key in enumerate(["N", "Среднее", "Медиана", "Ст. отклонение", "Top-2-box, %", "Bottom-2-box, %"]):
            cols[i].metric(key, summary.get(key, ""))
        nums = numeric_series(df[q]).dropna()
        if not nums.empty:
            dist = nums.value_counts().sort_index().reset_index()
            dist.columns = ["Оценка", "Количество"]
            fig = px.bar(dist, x="Оценка", y="Количество", title="Распределение оценок")
            st.plotly_chart(fig, use_container_width=True)
    elif qtype in ["single_choice", "multiple_choice"]:
        counts = categorical_counts(df[q], multi=(qtype == "multiple_choice"))
        st.dataframe(counts, use_container_width=True, hide_index=True)
        if not counts.empty:
            value_col = counts.columns[1]
            fig = px.bar(counts.head(30).iloc[::-1], x=value_col, y="Вариант", orientation="h", title="Распределение ответов")
            st.plotly_chart(fig, use_container_width=True)
    else:
        texts = df[q].dropna().astype(str).map(strip_html_tags)
        texts = texts[~texts.map(is_probably_junk_text)]
        st.metric("Непустых содержательных ответов", int(texts.shape[0]))
        st.dataframe(pd.DataFrame({"Ответ": texts}), use_container_width=True, hide_index=True)


def render_crosstabs_page():
    if not require_data():
        return
    st.header("Crosstabs")
    df = get_df()
    catalog = get_catalog()

    included = catalog.loc[catalog["include"], "column"].tolist()
    categorical = catalog.loc[catalog["include"] & catalog["type"].isin(["single_choice", "multiple_choice"]), "column"].tolist()
    rating = catalog.loc[catalog["include"] & catalog["type"].isin(["rating", "numeric"]), "column"].tolist()

    mode = st.radio("Тип анализа", ["Категория × категория", "Шкала × сегмент"], horizontal=True)

    if mode == "Категория × категория":
        if len(categorical) < 2:
            st.info("Нужно минимум два категориальных вопроса.")
            return
        c1, c2, c3 = st.columns([2, 2, 1])
        row_q = c1.selectbox("Строки", categorical, index=0)
        col_q = c2.selectbox("Колонки", categorical, index=1 if len(categorical) > 1 else 0)
        metric = c3.selectbox("Метрика", METRICS_CAT)
        table = crosstab_categorical(df, row_q, col_q, metric)
        st.session_state["last_crosstab"] = table
        if table.empty:
            st.warning("Нет данных для таблицы.")
            return
        st.dataframe(table, use_container_width=True)
        show_values = st.checkbox("Показывать числа на тепловой карте", value=True, key="cat_heatmap_values")
        fig = readable_heatmap(
            table,
            title=f"{shorten_label(row_q, 70)} × {shorten_label(col_q, 70)}: {metric}",
            value_name=metric,
            row_kind="Строка",
            col_kind="Колонка",
            prefer_question_numbers_on_y=False,
            max_x_label_chars=22,
            max_y_label_chars=42,
            show_values=show_values,
        )
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Расшифровка коротких подписей"):
            mapping = pd.concat([
                label_mapping_df(table.index.tolist(), "Строка", max_chars=42),
                label_mapping_df(table.columns.tolist(), "Колонка", max_chars=22),
            ], ignore_index=True)
            st.dataframe(mapping, use_container_width=True, hide_index=True)
    else:
        if not rating or not categorical:
            st.info("Нужен хотя бы один шкальный вопрос и один сегмент.")
            return
        c1, c2, c3 = st.columns([2, 2, 1])
        rating_q = c1.selectbox("Шкальный вопрос", rating)
        seg_q = c2.selectbox("Сегмент", categorical)
        metric = c3.selectbox("Метрика", METRICS_RATING)
        table = rating_by_segment(df, rating_q, seg_q, metric)
        st.session_state["last_crosstab"] = table
        st.dataframe(table, use_container_width=True)
        if not table.empty:
            fig = px.bar(table.reset_index(), x=metric, y=seg_q, orientation="h", title=f"{rating_q} по сегменту: {seg_q}")
            st.plotly_chart(fig, use_container_width=True)


def render_heatmaps_page():
    if not require_data():
        return
    st.header("Heatmaps")
    df = get_df()
    catalog = get_catalog()
    rating = catalog.loc[catalog["include"] & catalog["type"].isin(["rating", "numeric"]), "column"].tolist()
    categorical = catalog.loc[catalog["include"] & catalog["type"].isin(["single_choice", "multiple_choice"]), "column"].tolist()

    if not rating or not categorical:
        st.info("Для тепловой карты нужны шкальные вопросы и сегменты.")
        return

    heatmap_type = st.radio("Вид тепловой карты", ["Сегмент × шкальные вопросы", "Co-occurrence для multiple choice"], horizontal=True)

    if heatmap_type == "Сегмент × шкальные вопросы":
        seg_q = st.selectbox("Сегмент", categorical)
        metric = st.selectbox("Метрика", ["Среднее", "Top-2-box", "Bottom-2-box"])
        selected_ratings = st.multiselect("Шкальные вопросы", rating, default=rating[: min(12, len(rating))])
        rows = []
        for rq in selected_ratings:
            tmp = rating_by_segment(df, rq, seg_q, metric)
            if tmp.empty:
                continue
            row = tmp[metric].to_dict()
            row["Вопрос"] = rq
            rows.append(row)
        if not rows:
            st.warning("Нет данных для тепловой карты.")
            return
        matrix = pd.DataFrame(rows).set_index("Вопрос").fillna(np.nan)
        st.dataframe(matrix, use_container_width=True)
        c1, c2, c3 = st.columns([1, 1, 1])
        show_values = c1.checkbox("Показывать числа", value=True, key="segment_heatmap_values")
        compact_questions = c2.checkbox("На оси Y только номера вопросов", value=True, key="segment_heatmap_compact_y")
        chart_height = c3.slider("Высота графика", min_value=420, max_value=1100, value=max(460, min(900, 220 + 44 * len(matrix))), step=40)
        fig = readable_heatmap(
            matrix,
            title=f"{metric}: вопросы × {shorten_label(seg_q, 70)}",
            value_name=metric,
            row_kind="Вопрос",
            col_kind="Сегмент",
            prefer_question_numbers_on_y=compact_questions,
            max_x_label_chars=22,
            max_y_label_chars=46,
            show_values=show_values,
            height=chart_height,
        )
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Расшифровка коротких подписей"):
            mapping = pd.concat([
                label_mapping_df(matrix.index.tolist(), "Вопрос", max_chars=46, prefer_question_number=compact_questions),
                label_mapping_df(matrix.columns.tolist(), "Сегмент", max_chars=22),
            ], ignore_index=True)
            st.dataframe(mapping, use_container_width=True, hide_index=True)
    else:
        multi_cols = catalog.loc[catalog["include"] & catalog["type"].eq("multiple_choice"), "column"].tolist()
        if not multi_cols:
            st.info("Нет multiple choice вопросов.")
            return
        q = st.selectbox("Multiple choice вопрос", multi_cols)
        options_per_row = df[q].apply(split_multi)
        all_options = sorted({opt for opts in options_per_row for opt in opts})
        if len(all_options) < 2:
            st.info("Недостаточно вариантов для co-occurrence.")
            return
        selected_options = st.multiselect("Варианты", all_options, default=all_options[: min(20, len(all_options))])
        data = pd.DataFrame(0, index=selected_options, columns=selected_options, dtype=int)
        for opts in options_per_row:
            opts_set = set(opts) & set(selected_options)
            for a in opts_set:
                for b in opts_set:
                    data.loc[a, b] += 1
        st.dataframe(data, use_container_width=True)
        show_values = st.checkbox("Показывать числа", value=True, key="cooccurrence_values")
        fig = readable_heatmap(
            data,
            title=f"Co-occurrence: {shorten_label(q, 80)}",
            value_name="Количество",
            row_kind="Вариант",
            col_kind="Вариант",
            prefer_question_numbers_on_y=False,
            max_x_label_chars=24,
            max_y_label_chars=36,
            show_values=show_values,
        )
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Расшифровка коротких подписей"):
            mapping = pd.concat([
                label_mapping_df(data.index.tolist(), "Строка", max_chars=36),
                label_mapping_df(data.columns.tolist(), "Колонка", max_chars=24),
            ], ignore_index=True)
            st.dataframe(mapping, use_container_width=True, hide_index=True)


def render_drivers_page():
    if not require_data():
        return
    st.header("Корреляции и драйверы")
    df = get_df()
    catalog = get_catalog()
    rating = catalog.loc[catalog["include"] & catalog["type"].isin(["rating", "numeric"]), "column"].tolist()
    if len(rating) < 2:
        st.info("Нужно минимум два шкальных/числовых вопроса.")
        return

    nums = df[rating].apply(numeric_series)
    st.subheader("Корреляционная карта")
    method = st.selectbox("Метод корреляции", ["spearman", "pearson"], index=0)
    corr = nums.corr(method=method).round(2)
    st.dataframe(corr, use_container_width=True)
    fig = readable_heatmap(
        corr,
        title=f"Корреляции ({method})",
        value_name="Корреляция",
        row_kind="Вопрос",
        col_kind="Вопрос",
        prefer_question_numbers_on_y=True,
        max_x_label_chars=18,
        max_y_label_chars=34,
        show_values=True,
    )
    fig.update_traces(colorscale="RdBu", zmin=-1, zmax=1)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Расшифровка коротких подписей"):
        mapping = pd.concat([
            label_mapping_df(corr.index.tolist(), "Строка", max_chars=34, prefer_question_number=True),
            label_mapping_df(corr.columns.tolist(), "Колонка", max_chars=18, prefer_question_number=True),
        ], ignore_index=True)
        st.dataframe(mapping, use_container_width=True, hide_index=True)

    st.subheader("Driver analysis")
    outcome_candidates = catalog.loc[catalog["include"] & catalog["type"].isin(["rating", "numeric"]), "column"].tolist()
    outcome = st.selectbox("Итоговая метрика / outcome", outcome_candidates)
    drivers = []
    for q in rating:
        if q == outcome:
            continue
        pair = nums[[outcome, q]].dropna()
        if pair.shape[0] < 10:
            continue
        corr_val = pair[outcome].corr(pair[q], method="spearman")
        summary = rating_summary(df[q])
        drivers.append(
            {
                "Драйвер": q,
                "Связь с outcome": round(float(corr_val), 3) if pd.notna(corr_val) else np.nan,
                "N": int(pair.shape[0]),
                "Текущая оценка": summary.get("Среднее"),
                "Top-2-box, %": summary.get("Top-2-box, %"),
                "Bottom-2-box, %": summary.get("Bottom-2-box, %"),
            }
        )
    drivers_df = pd.DataFrame(drivers).sort_values("Связь с outcome", ascending=False)
    st.session_state["drivers"] = drivers_df
    st.dataframe(drivers_df, use_container_width=True, hide_index=True)

    if not drivers_df.empty:
        plot_df = drivers_df.dropna(subset=["Связь с outcome", "Текущая оценка"])
        if not plot_df.empty:
            fig2 = px.scatter(
                plot_df,
                x="Текущая оценка",
                y="Связь с outcome",
                size="N",
                hover_name="Драйвер",
                title="Priority matrix: оценка × связь с outcome",
            )
            st.plotly_chart(fig2, use_container_width=True)


def render_open_text_page():
    if not require_data():
        return
    st.header("Открытые ответы")
    df = get_df()
    catalog = get_catalog()
    text_cols = catalog.loc[catalog["include"] & catalog["type"].isin(["open_text", "other_text"]), "column"].tolist()
    categorical = catalog.loc[catalog["include"] & catalog["type"].isin(["single_choice", "multiple_choice"]), "column"].tolist()
    if not text_cols:
        st.info("Открытые вопросы не найдены.")
        return
    q = st.selectbox("Открытый вопрос", text_cols)
    texts = df[q].dropna().astype(str).map(strip_html_tags)
    texts = texts[~texts.map(is_probably_junk_text)]
    st.metric("Содержательных ответов", int(texts.shape[0]))
    st.dataframe(pd.DataFrame({"Ответ": texts}), use_container_width=True, hide_index=True, height=300)

    st.subheader("Частотность слов")
    wc = build_open_text_word_counts(df[q])
    st.dataframe(wc.head(50), use_container_width=True, hide_index=True)
    if not wc.empty:
        fig = px.bar(wc.head(30).iloc[::-1], x="Количество", y="Слово", orientation="h", title="Самые частые слова")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Простая ручная кодировка по ключевым словам")
    st.caption("Введите темы и слова через запятую. Например: документация: инструкция, справка, пример")
    default_codes = "документация: инструкция, справка, пример\nошибки: ошибка, баг, вылет, завис\nпеременные: переменн, csv, импорт, экспорт\nвизуализация: экран, hmi, визуал"
    code_text = st.text_area("Кодбук", value=default_codes, height=140)
    codes: Dict[str, List[str]] = {}
    for line in code_text.splitlines():
        if ":" in line:
            name, kws = line.split(":", 1)
            keywords = [k.strip().lower() for k in kws.split(",") if k.strip()]
            if name.strip() and keywords:
                codes[name.strip()] = keywords
    if codes:
        coded_rows = []
        for idx, text in texts.items():
            lower = text.lower()
            matched = [name for name, kws in codes.items() if any(k in lower for k in kws)]
            if not matched:
                matched = ["Без темы"]
            for name in matched:
                coded_rows.append({"row": idx, "Тема": name, "Ответ": text})
        coded = pd.DataFrame(coded_rows)
        freq = coded["Тема"].value_counts().reset_index()
        freq.columns = ["Тема", "Количество"]
        st.dataframe(freq, use_container_width=True, hide_index=True)
        st.session_state["open_text_codes"] = coded

        if categorical:
            seg_q = st.selectbox("Сравнить темы по сегменту", ["Не сравнивать"] + categorical)
            if seg_q != "Не сравнивать":
                seg = explode_question(df, seg_q).rename(columns={"__row_id": "row"})
                merged = coded.merge(seg, on="row", how="inner")
                if not merged.empty:
                    tab = pd.crosstab(merged["Тема"], merged[seg_q])
                    st.dataframe(tab, use_container_width=True)
                    fig = readable_heatmap(
                        tab,
                        title=f"Темы × {shorten_label(seg_q, 70)}",
                        value_name="Количество",
                        row_kind="Тема",
                        col_kind="Сегмент",
                        prefer_question_numbers_on_y=False,
                        max_x_label_chars=24,
                        max_y_label_chars=38,
                        show_values=True,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    with st.expander("Расшифровка коротких подписей"):
                        mapping = pd.concat([
                            label_mapping_df(tab.index.tolist(), "Тема", max_chars=38),
                            label_mapping_df(tab.columns.tolist(), "Сегмент", max_chars=24),
                        ], ignore_index=True)
                        st.dataframe(mapping, use_container_width=True, hide_index=True)


def render_export_page():
    if not require_data():
        return
    st.header("Экспорт")
    df = get_df()
    catalog = get_catalog()
    tables = {
        "Raw data": df,
        "Question catalog": catalog,
        "Topline": st.session_state.get("topline", build_topline_table(df, catalog)),
    }
    if "last_crosstab" in st.session_state:
        tables["Last crosstab"] = st.session_state["last_crosstab"]
    if "drivers" in st.session_state:
        tables["Drivers"] = st.session_state["drivers"]
    if "open_text_codes" in st.session_state:
        tables["Open text codes"] = st.session_state["open_text_codes"]

    excel_bytes = export_dataframes_to_excel(tables)
    st.download_button(
        "Скачать Excel-отчёт",
        data=excel_bytes,
        file_name="survey_analysis_export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )

    st.write("В экспорт попадут: исходные данные, словарь вопросов, topline, последний построенный crosstab, драйверы и коды открытых ответов, если они уже построены.")


def main():
    page = render_sidebar()
    if page == "1. Загрузка":
        render_upload_page()
    elif page == "2. Словарь вопросов":
        render_catalog_page()
    elif page == "3. Качество данных":
        render_quality_page()
    elif page == "4. Topline":
        render_topline_page()
    elif page == "5. Crosstabs":
        render_crosstabs_page()
    elif page == "6. Heatmaps":
        render_heatmaps_page()
    elif page == "7. Корреляции и драйверы":
        render_drivers_page()
    elif page == "8. Открытые ответы":
        render_open_text_page()
    elif page == "9. Экспорт":
        render_export_page()


if __name__ == "__main__":
    main()
