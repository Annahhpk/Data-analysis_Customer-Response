import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


INFILE = "D:\\Настя\\Мои проекты\\Аналитика данных\\data\\data_set.xlsx"
OUTDIR = "D:\\Настя\\Мои проекты\\Аналитика данных\\data\\ads_response_outputs"
# CLEANED_FILE = os.path.join(os.path.dirname(OUTDIR), "data_set_cleaned.xlsx")
CLEANED_FILE = os.path.join(OUTDIR, "data_set_cleaned.xlsx")


# -----------------------------
# Очистка данных (вставлено как просила)
# -----------------------------
def clean_dataset(df):
    # 1. Удаление неинформативных признаков
    columns_for_drop = [
        'PREVIOUS_CARD_NUM_UTILIZED', 'LOAN_DLQ_NUM', 'FL_PRESENCE_FL', 'OWN_AUTO',
        'AUTO_RUS_FL', 'HS_PRESENCE_FL', 'COT_PRESENCE_FL', 'GAR_PRESENCE_FL',
        'LAND_PRESENCE_FL', 'REGION_NM', 'AGREEMENT_RK', 'ORG_TP_FCAPITAL',
        'JOB_DIR', 'CREDIT', 'FST_PAYMENT', 'REG_ADDRESS_PROVINCE',
        'POSTAL_ADDRESS_PROVINCE', 'TP_PROVINCE', 'REG_FACT_FL', 'FACT_POST_FL',
        'REG_POST_FL', 'REG_FACT_POST_FL', 'REG_FACT_POST_TP_FL', 'TERM',
        'DL_DOCUMENT_FL', 'GPF_DOCUMENT_FL', 'FACT_LIVING_TERM', 'WORK_TIME',
        'FACT_PHONE_FL', 'REG_PHONE_FL', 'GEN_PHONE_FL', 'LOAN_NUM_TOTAL',
        'LOAN_NUM_CLOSED', 'LOAN_NUM_PAYM', 'LOAN_MAX_DLQ', 'LOAN_AVG_DLQ_AMT',
        'LOAN_MAX_DLQ_AMT'
    ]
    df.drop(columns=[col for col in columns_for_drop if col in df.columns], inplace=True)

    # 2. Заполнение пропусков
    for col in df.columns:
        if df[ col ].dtype == 'object' or str(df[ col ].dtype).startswith('category'):
            # Заполняем модой (mode()[0]) для категориальных признаков
            df[ col ] = df[ col ].fillna(df[ col ].mode()[ 0 ])
        else:
            # Заполняем медианой для числовых признаков
            df[ col ] = df[ col ].fillna(df[ col ].median())

    return df


# -----------------------------
# Вспомогательные
# -----------------------------
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def weighted_mean(series, weights):
    w = np.asarray(weights)
    x = pd.to_numeric(series, errors="coerce").to_numpy()
    mask = ~np.isnan(x) & ~np.isnan(w)
    if mask.sum() == 0:
        return np.nan
    return np.average(x[mask], weights=w[mask])

def weighted_quantile(values, quantiles, sample_weight=None):
    """Взвешенные квантили; quantiles — список долей (0..1)."""
    v = pd.to_numeric(values, errors="coerce").to_numpy()
    q = np.asarray(quantiles)
    if sample_weight is None:
        sw = np.ones(len(v))
    else:
        sw = np.asarray(sample_weight)
    mask = ~np.isnan(v) & ~np.isnan(sw)
    if mask.sum() == 0:
        return [np.nan for _ in q]
    v = v[mask]
    sw = sw[mask]
    sorter = np.argsort(v)
    v = v[sorter]
    sw = sw[sorter]
    cdf = np.cumsum(sw) - 0.5 * sw
    cdf /= np.sum(sw)
    return np.interp(q, cdf, v)

def chi2_pvalue_categorical(col, y):
    s = col.astype(str).fillna("Unknown")
    # защита от чрезмерной разреженности
    top_cats = s.value_counts().nlargest(50).index
    s = np.where(np.isin(s, top_cats), s, "Other")
    ct = pd.crosstab(s, y)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 1.0
    chi2, p, _, _ = stats.chi2_contingency(ct)
    return p

def ttest_pvalue_numeric(col, y):
    a = pd.to_numeric(col[y == 0], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    b = pd.to_numeric(col[y == 1], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(a) < 5 or len(b) < 5:
        return 1.0
    _, p = stats.ttest_ind(a, b, equal_var=False)
    return p

def weighted_top(series, weights, top_n):
    """Возвращает список (значение, процент) топ-N категорий по сумме весов."""
    s = series.astype(str).fillna("Unknown").replace({"nan": "Unknown"})
    w = pd.Series(weights, index=series.index)
    agg = pd.DataFrame({"val": s, "w": w}).groupby("val")["w"].sum().sort_values(ascending=False)
    total = agg.sum()
    if total <= 0:
        return []
    result = [(k, round(agg.loc[k] / total * 100, 1)) for k in agg.index[:top_n]]
    return result

def fmt_top_list(pairs):
    """Печать как 'A (12.3%); B (9.1%); ...'"""
    return "; ".join([f"{name} ({pct}%)" for name, pct in pairs])


def main():
    ensure_dir(OUTDIR)
    # 1) Загрузка
    df_raw = pd.read_excel(INFILE)
    df_raw.drop_duplicates(inplace=True)

    # Проверим наличие целевой и ID в сыром файле (до очистки)
    if "TARGET" not in df_raw.columns:
        raise ValueError("В файле нет колонки TARGET")
    if "AGREEMENT_RK" not in df_raw.columns:
        raise ValueError("В файле нет колонки AGREEMENT_RK")

    target = "TARGET"
    id_col = "AGREEMENT_RK"

    # --- Сохраним ID до очистки, т.к. clean_dataset удаляет AGREEMENT_RK ---
    id_series = df_raw[id_col].copy()

    # 1.1) Очистка
    df = clean_dataset(df_raw.copy())

    # Вернём AGREEMENT_RK в датафрейм, если его удалили
    if id_col not in df.columns:
        df[id_col] = id_series

    # Сохраняем очищенный датасет
    df.to_excel(CLEANED_FILE, index=False)
    print(f"✅ Очищенный датасет сохранён: {CLEANED_FILE}")

    # 2) Определение типов и базовая импьютация для отбора признаков (ручные t-test/χ²)
    cat_cols = [c for c in df.columns if c not in [target, id_col] and (df[c].dtype == "object" or str(df[c].dtype).startswith("category"))]
    num_cols = [c for c in df.columns if c not in [target, id_col] and c not in cat_cols]

    df_imp = df.copy()
    for c in num_cols:
        med = pd.to_numeric(df_imp[c], errors="coerce").median(skipna=True)
        df_imp[c] = pd.to_numeric(df_imp[c], errors="coerce").fillna(med)
    for c in cat_cols:
        df_imp[c] = df_imp[c].astype(str).fillna("Unknown").replace({"nan": "Unknown"})

    y = df_imp[target].astype(int)

    # 3) Отбор признаков (t-test/χ²)
    selected_num = []
    for c in num_cols:
        try:
            if ttest_pvalue_numeric(df_imp[c], y) < 0.05:
                selected_num.append(c)
        except Exception:
            pass

    selected_cat = []
    for c in cat_cols:
        try:
            if chi2_pvalue_categorical(df_imp[c], y) < 0.05:
                selected_cat.append(c)
        except Exception:
            pass

    # Фоллбэки
    if len(selected_num) == 0 and len(num_cols) > 0:
        selected_num = num_cols[:min(5, len(num_cols))]
    if len(selected_cat) == 0 and len(cat_cols) > 0:
        selected_cat = cat_cols[:min(5, len(cat_cols))]

    # 4) Трейн/тест
    X = df[selected_num + selected_cat + [id_col]].copy()
    y = df[target].astype(int).copy()
    X_features = X.drop(columns=[id_col])

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_features, y, X.index, test_size=0.3, stratify=y, random_state=42
    )

    # 5) Препроцессинг + модель (OHE для категориальных, sparse_output=False — как в исходнике)
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocess = ColumnTransformer([
        ("num", num_pipe, [c for c in selected_num if c in X_features.columns]),
        ("cat", cat_pipe, [c for c in selected_cat if c in X_features.columns])
    ], remainder="drop")

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    pipe = Pipeline([("prep", preprocess), ("clf", rf)])

    param_grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [5, 10, None],
        "clf__min_samples_split": [2, 5],
        "clf__class_weight": ["balanced", {0: 1, 1: 3}]
    }

    grid = GridSearchCV(pipe, param_grid=param_grid, cv=3, scoring="f1", n_jobs=-1, refit=True)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # 6) Оценка и порог
    y_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred_default = best_model.predict(X_test)

    thresholds = np.linspace(0.1, 0.9, 81)
    f1_scores = [f1_score(y_test, (y_proba >= t).astype(int), zero_division=0) for t in thresholds]
    best_threshold = float(thresholds[int(np.argmax(f1_scores))])

    roc_auc = roc_auc_score(y_test, y_proba)
    f1_def = f1_score(y_test, y_pred_default)

    # 7) Таблица теста и отбор клиентов
    test_df = df.loc[idx_test, [id_col, target] + selected_num + selected_cat].copy()
    test_df["PREDICTED_PROB"] = y_proba
    test_df["PREDICTED"] = (y_proba >= best_threshold).astype(int)

    high_prob_clients = test_df[test_df["PREDICTED_PROB"] >= best_threshold].copy()
    high_prob_ids = high_prob_clients[id_col].unique().tolist()

    ensure_dir(OUTDIR)
    top_clients = high_prob_clients[[id_col, "PREDICTED_PROB"]].sort_values("PREDICTED_PROB", ascending=False).head(50)
    top_clients.to_csv(os.path.join(OUTDIR, "top_50_clients.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame({id_col: high_prob_ids}).to_csv(os.path.join(OUTDIR, "all_high_prob_clients.csv"), index=False, encoding="utf-8-sig")

    # 8) Важность признаков в пространстве OHE
    prep = best_model.named_steps["prep"]
    clf = best_model.named_steps["clf"]

    feature_names = []
    if "num" in prep.named_transformers_ and prep.named_transformers_["num"] != "drop":
        nums = prep.transformers_[0][2]
        feature_names.extend(nums)
    if "cat" in prep.named_transformers_ and prep.named_transformers_["cat"] != "drop":
        cat_cols_used = prep.transformers_[1][2]
        ohe = prep.named_transformers_["cat"].named_steps["ohe"]
        ohe_names = ohe.get_feature_names_out(cat_cols_used).tolist()
        feature_names.extend(ohe_names)

    importances = clf.feature_importances_
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    fi.to_csv(os.path.join(OUTDIR, "feature_importances.csv"), header=["importance"], encoding="utf-8-sig")

    plt.figure(figsize=(10, 6))
    fi.head(30).sort_values().plot(kind="barh")
    plt.title("Важность признаков (топ-30)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "feature_importances.png"))
    plt.close()

    # 9) Распределение вероятностей
    plt.figure(figsize=(10, 6))
    plt.hist(y_proba, bins=40)
    plt.axvline(best_threshold)
    plt.title("Распределение вероятностей отклика (тест)")
    plt.xlabel("Вероятность положительного отклика")
    plt.ylabel("Количество клиентов")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "probability_distribution.png"))
    plt.close()

    # 10) Портрет (взвешенно по PREDICTED_PROB), с требуемыми правилами вывода
    portrait_source = high_prob_clients.copy()
    if portrait_source.empty:
        portrait_source = test_df.copy()
    weights = portrait_source["PREDICTED_PROB"].clip(lower=0)

    # --- Числовые ---
    portrait_lines = []
    # AGE — среднее + P10/P90
    if "AGE" in portrait_source.columns:
        age_mean = weighted_mean(portrait_source["AGE"], weights)
        age_p10, age_p90 = weighted_quantile(portrait_source["AGE"], [0.1, 0.9], sample_weight=weights)
        portrait_lines.append(f"• AGE: {age_mean:.1f} (P10={age_p10:.1f}; P90={age_p90:.1f})")

    # PERSONAL_INCOME — среднее + P10/P90
    if "PERSONAL_INCOME" in portrait_source.columns:
        inc_mean = weighted_mean(portrait_source["PERSONAL_INCOME"], weights)
        inc_p10, inc_p90 = weighted_quantile(portrait_source["PERSONAL_INCOME"], [0.1, 0.9], sample_weight=weights)
        portrait_lines.append(f"• PERSONAL_INCOME: {inc_mean:.1f} (P10={inc_p10:.1f}; P90={inc_p90:.1f})")

    # CHILD_TOTAL — как есть
    if "CHILD_TOTAL" in portrait_source.columns:
        ch_mean = weighted_mean(portrait_source["CHILD_TOTAL"], weights)
        ch_p10, ch_p90 = weighted_quantile(portrait_source["CHILD_TOTAL"], [0.1, 0.9], sample_weight=weights)
        portrait_lines.append(f"• CHILD_TOTAL: {ch_mean:.1f} (P10={ch_p10:.1f}; P90={ch_p90:.1f})")

    # DEPENDANTS — только целые (медиана + P10/P90 округляем)
    if "DEPENDANTS" in portrait_source.columns:
        dep_med = weighted_quantile(portrait_source["DEPENDANTS"], [0.5], sample_weight=weights)[0]
        dep_p10, dep_p90 = weighted_quantile(portrait_source["DEPENDANTS"], [0.1, 0.9], sample_weight=weights)
        portrait_lines.append(f"• DEPENDANTS: {int(round(dep_med))} (P10={int(round(dep_p10))}; P90={int(round(dep_p90))})")

    # --- Категориальные ---
    if "GENDER" in portrait_source.columns:
        gender_top = weighted_top(portrait_source["GENDER"], weights, top_n=1)
        if gender_top:
            portrait_lines.append(f"• GENDER: {gender_top[0][0]} ({gender_top[0][1]}%)")

    if "EDUCATION" in portrait_source.columns:
        edu_top = weighted_top(portrait_source["EDUCATION"], weights, top_n=3)
        if edu_top:
            portrait_lines.append("• EDUCATION: " + fmt_top_list(edu_top))

    if "MARITAL_STATUS" in portrait_source.columns:
        mar_top = weighted_top(portrait_source["MARITAL_STATUS"], weights, top_n=2)
        if mar_top:
            portrait_lines.append("• MARITAL_STATUS: " + fmt_top_list(mar_top))

    if "GEN_INDUSTRY" in portrait_source.columns:
        ind_top = weighted_top(portrait_source["GEN_INDUSTRY"], weights, top_n=5)
        if ind_top:
            portrait_lines.append("• GEN_INDUSTRY: " + fmt_top_list(ind_top))

    if "GEN_TITLE" in portrait_source.columns:
        title_top = weighted_top(portrait_source["GEN_TITLE"], weights, top_n=3)
        if title_top:
            portrait_lines.append("• GEN_TITLE: " + fmt_top_list(title_top))

    if "ORG_TP_STATE" in portrait_source.columns:
        org_top = weighted_top(portrait_source["ORG_TP_STATE"], weights, top_n=3)
        if org_top:
            portrait_lines.append("• ORG_TP_STATE: " + fmt_top_list(org_top))

    if "FAMILY_INCOME" in portrait_source.columns:
        finc_top = weighted_top(portrait_source["FAMILY_INCOME"], weights, top_n=1)
        if finc_top:
            portrait_lines.append(f"• FAMILY_INCOME: {finc_top[0][0]} ({finc_top[0][1]}%)")

    if "FACT_ADDRESS_PROVINCE" in portrait_source.columns:
        prov_top = weighted_top(portrait_source["FACT_ADDRESS_PROVINCE"], weights, top_n=5)
        if prov_top:
            portrait_lines.append("• FACT_ADDRESS_PROVINCE: " + fmt_top_list(prov_top))

    # Сохраняем портрет в csv
    pd.DataFrame({"portrait_line": portrait_lines}).to_csv(
        os.path.join(OUTDIR, "client_portrait.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # Визуализации
    if "GENDER" in portrait_source.columns:
        plt.figure(figsize=(6, 4))
        portrait_source.assign(w=weights).groupby("GENDER")["w"].sum().sort_values(ascending=False).plot(kind="bar")
        plt.title("Распределение по полу (взвешенно)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "gender_distribution.png"))
        plt.close()

    if "AGE" in portrait_source.columns:
        plt.figure(figsize=(6, 4))
        plt.hist(portrait_source["AGE"], weights=weights, bins=20)
        plt.title("Распределение по возрасту (взвешенно)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "age_distribution.png"))
        plt.close()

    if "PERSONAL_INCOME" in portrait_source.columns:
        plt.figure(figsize=(6, 4))
        plt.hist(portrait_source["PERSONAL_INCOME"], weights=weights, bins=20)
        plt.title("Распределение личного дохода (взвешенно)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "income_distribution.png"))
        plt.close()

    # 11) Итоговый отчёт
    report_lines = []
    report_lines.append("=" * 90)
    report_lines.append("АНАЛИТИЧЕСКИЙ ОТЧЁТ")
    report_lines.append("=" * 90)
    report_lines.append(f"Выбранные числовые признаки (t-test p<0.05): {selected_num}")
    report_lines.append(f"Выбранные категориальные признаки (chi2 p<0.05): {selected_cat}")
    report_lines.append(f"Лучшие параметры модели: {grid.best_params_}")
    report_lines.append(f"Оптимальный порог вероятности: {best_threshold:.2f}")
    report_lines.append(f"Отобрано клиентов: {len(high_prob_ids)} из {len(X_test)} (тест)")
    report_lines.append(f"AUC-ROC: {roc_auc:.4f}")
    report_lines.append(f"F1 (порог=0.5): {f1_def:.4f}")
    report_lines.append("\nОтчёт по метрикам классификации (порог=0.5):")
    report_lines.append(classification_report(y_test, y_pred_default, digits=4))

    report_lines.append("\nИТОГОВЫЙ ВЫВОД ДЛЯ БИЗНЕСА")
    report_lines.append(f"- Рекомендуется обращаться к клиентам с вероятностью отклика ≥ {best_threshold:.2f}.")
    report_lines.append(f"- В тестовой выборке таких клиентов: {len(high_prob_ids)}.")
    report_lines.append("- Провести A/B тестирование порога и персонализировать оффер под портрет ниже.")

    report_lines.append("\nФИНАЛЬНОЕ РЕЗЮМЕ: Портрет клиента (взвешенно по вероятности):")
    report_lines.extend(portrait_lines)

    with open(os.path.join(OUTDIR, 'business_report.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))

    print("\n".join(report_lines[:25]))
    print("...")
    print(f"Готово. Файлы сохранены в: {OUTDIR}")
    print(f"Очищенный датасет сохранён в: {CLEANED_FILE}")

if __name__ == "__main__":
    main()

