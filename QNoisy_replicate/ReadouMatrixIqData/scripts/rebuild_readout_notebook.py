"""Rebuild readout_matrix.ipynb without inline function definitions."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).resolve().parent.parent / "readout_matrix.ipynb"


def _cell(cell_type: str, source: str) -> dict:
    return {
        "cell_type": cell_type,
        "id": str(uuid.uuid4())[:8],
        "metadata": {},
        "source": [line + "\n" for line in source.splitlines()],
    }


def _code(source: str) -> dict:
    cell = _cell("code", source)
    cell["execution_count"] = None
    cell["outputs"] = []
    return cell


def build_cells() -> list[dict]:
    return [
        _cell(
            "markdown",
            """# Part 1 — استخراج ماتریس نویز خوانش از داده IQ

**هدف:** از داده‌های اندازه‌گیری تجربی (ابرهای IQ)، ماتریس‌های نویز خوانش ساخته می‌شوند تا در شبیه‌سازی مدار و تولید دیتاست ML (Part 2) استفاده شوند.

**خروجی اصلی:**
- ماتریس انتساب تک‌کیوبیتی ۲×۲ برای هر کیوبیت (`assignment_matrix_q1` … `q4`)
- ماتریس همبسته ۱۶×۱۶ (`noise_matrix_16x16`)

**پیاده‌سازی:** همه توابع در `src/readout_extraction/` هستند؛ این notebook فقط فراخوانی و بررسی بصری انجام می‌دهد.

**جایگزین CLI (بدون notebook):**
```bash
python scripts/run_readout_matrix.py --list
python scripts/run_readout_matrix.py --snapshot 1_1_2025
```

**مرحله بعد (Part 2):** پس از ذخیره ماتریس‌ها → `python scripts/build_ml_datasets.py --snapshot 1_1_2025`""",
        ),
        _cell(
            "markdown",
            """## ساختار داده و مسیرها

```
quadrature_data_4qubits/     ← ورودی (داده IQ تجربی)
    1_1_2025/                ← snapshot (یک کمپین اندازه‌گیری)
        0000.txt … 1111.txt  ← ۱۶ حالت پایه، هر فایل ۴ خط × ۱۰۰۰ نمونه IQ

noise_matrix_results/        ← خروجی این notebook
    res_1_1_2025/            ← متناظر با همان snapshot
        assignment_matrix_q1.txt … q4.txt
        noise_matrix_16x16.txt
        (+ نمودارها و .npy اختیاری)
```

| متغیر | نقش |
|--------|-----|
| `CAMPAIGN` | پوشه کمپین (`quadrature_data_4qubits`) |
| `SNAPSHOT` | زیرپوشه تاریخ/نسخه (مثلاً `1_1_2025`) |
| `config.data_dir` | مسیر خواندن فایل‌های `.txt` |
| `config.output_dir` | مسیر ذخیره ماتریس‌ها (`res_<snapshot>`) |

**ترتیب اجرا:** ابتدا سلول **Setup** را اجرا کنید؛ سپس یا سلول «خط لوله کامل» یا مراحل تکی.""",
        ),
        _code(
            """# --- Setup: imports + config + load data ---
import numpy as np

from src.readout_extraction import (
    ReadoutDatasetConfig,
    load_all_data,
    train_all_classifiers,
    predict_full_dataset,
    build_noise_matrix,
    validate_noise_matrix,
    save_noise_matrix,
    run_noise_extraction,
    build_independent_noise_matrix,
    compare_noise_models,
    plot_fourqubit_state_overlay,
    plot_qubit_iq_clouds,
    plot_noise_matrix,
    plot_matrix_difference,
    run_readout_pipeline,
)

CAMPAIGN = "./quadrature_data_4qubits"
SNAPSHOT = "1_1_2025"   # ← snapshot مورد نظر را اینجا عوض کنید

config = ReadoutDatasetConfig.from_snapshot(CAMPAIGN, SNAPSHOT)
config.classifier_type = "LDA"
config.qda_reg_param = 0.3

X, y, shots_per_file = load_all_data(config)
print(f"Loaded {X.shape[0]} shots from {config.data_dir}")
print(f"Results -> {config.output_dir}")"""
        ),
        _cell(
            "markdown",
            """### Setup — چه کاری انجام شد؟

- `ReadoutDatasetConfig.from_snapshot` مسیر ورودی/خروجی را از `CAMPAIGN` و `SNAPSHOT` می‌سازد.
- `load_all_data` همه ۱۶ فایل را می‌خواند → `X` شکل `(16000, 4)` (مقادیر مختلط IQ)، `y` برچسب آماده‌شده.
- خروجی ماتریس‌ها در `noise_matrix_results/res_<SNAPSHOT>/` ذخیره خواهد شد.""",
        ),
        _cell(
            "markdown",
            """### اجرای کامل خط لوله (پیشنهادی)

یک‌جا: آموزش classifier → ساخت ۱۶×۱۶ → اعتبارسنجی → ذخیره → نمودار → مقایسه با مدل مستقل Kronecker.

معادل: `run_readout_pipeline(config)` در `src/readout_extraction/pipeline.py`.""",
        ),
        _code(
            """# همه مراحل: train -> matrix -> validate -> save -> plots -> comparison
results = run_readout_pipeline(config)
noise_matrix = results["noise_matrix"]
assignment_matrices = results["assignment_matrices"]"""
        ),
        _cell(
            "markdown",
            """### مرحله ۱ — بررسی نمونه داده

نمایش یک شات: مقدار IQ هر کیوبیت و برچسب آماده‌شده (`0` یا `1`).""",
        ),
        _code("X[0], y[0]"),
        _cell(
            "markdown",
            """### مرحله ۲ — نمودار overlay ابرهای IQ

**وظیفه:** تفکیک بصری حالت‌های اندازه‌گیری برای حالت‌های پایه انتخابی (مثلاً `1010` و `0101`).

**داده:** همان `X, y` بارگذاری‌شده از snapshot فعال.""",
        ),
        _code(
            """plot_fourqubit_state_overlay(
    X, y, config,
    target_states=("1010", "0101"),
    show=True,
)"""
        ),
        _cell(
            "markdown",
            """### مرحله ۳ — آموزش classifier تک‌کیوبیتی

**مدل:** Linear Discriminant Analysis (LDA) روی ویژگی `[I, Q]` برای هر کیوبیت.

**خروجی:**
- `assignment_matrices` — ماتریس ۲×۲ per qubit
- `y_pred` — پیش‌بینی برای کل ۱۶۰۰۰ شات (ورودی ساخت ماتریس ۱۶×۱۶)""",
        ),
        _code(
            """classifiers, scalers, assignment_matrices = train_all_classifiers(X, y, config)
y_pred = predict_full_dataset(X, classifiers, scalers, config.num_qubits)"""
        ),
        _cell(
            "markdown",
            """### مرحله ۴ — ساخت ماتریس نویز ۱۶×۱۶

**تعریف:** `M[i,j] = P(measured=j | prepared=i)` برای ۱۶ حالت پایه.

**ورودی:** برچسب واقعی `y` و پیش‌بینی `y_pred` از مرحله قبل.""",
        ),
        _code(
            """noise_matrix = build_noise_matrix(y, y_pred, config.num_qubits)
fidelity = noise_matrix.diagonal().mean()
print(f"Average assignment fidelity = {fidelity:.6f}")"""
        ),
        _cell(
            "markdown",
            """### مرحله ۵ — اعتبارسنجی و ذخیره

- بررسی جمع سطرها (= ۱) و fidelity قطری
- ذخیره در `config.output_dir` (فایل‌های `.txt` و در صورت نیاز `.npy`)

این فایل‌ها توسط `src/noises.py` در Part 2 خوانده می‌شوند.""",
        ),
        _code(
            """validate_noise_matrix(noise_matrix)
save_noise_matrix(noise_matrix, config.output_dir, config.num_qubits)"""
        ),
        _cell(
            "markdown",
            """### مرحله ۶ — heatmap ماتریس نویز

نمایش ماتریس ۱۶×۱۶ برای بررسی همبستگی و crosstalk خوانش.""",
        ),
        _code(
            """plot_noise_matrix(
    noise_matrix,
    config,
    show=True,
)"""
        ),
        _cell(
            "markdown",
            """### مرحله ۷ — مقایسه LDA و QDA

دو classifier روی همان داده؛ مقایسه fidelity قطری و اختلاف Frobenius.""",
        ),
        _code(
            """M_LDA = run_noise_extraction(X, y, config, classifier_type="LDA")
M_QDA = run_noise_extraction(X, y, config, classifier_type="QDA", qda_reg_param=0.30)

print("Mean diag LDA:", M_LDA.diagonal().mean())
print("Mean diag QDA:", M_QDA.diagonal().mean())
print("Frobenius norm:", np.linalg.norm(M_LDA - M_QDA))
print("Max abs diff:", np.max(np.abs(M_LDA - M_QDA)))"""
        ),
        _cell(
            "markdown",
            """### مرحله ۸ — ابر IQ تک‌کیوبیتی

جداسازی بصری حالت‌های |0⟩ و |1⟩ برای یک کیوبیت (`qubit_index=0` → کیوبیت ۱).""",
        ),
        _code("plot_qubit_iq_clouds(X, y, qubit_index=0, show=True)"),
        _cell(
            "markdown",
            """### مرحله ۹ — مقایسه مدل همبسته و مستقل (Kronecker)

**همبسته (`A_real`):** ماتریس ۱۶×۱۶ استخراج‌شده از داده.

**مستقل (`A_indep`):** حاصل ضرب کرونکر ماتریس‌های ۲×۲ تک‌کیوبیتی.

اختلاف این دو میزان crosstalk و نویز غیرقابل فاکتورسازی را نشان می‌دهد.""",
        ),
        _code(
            """A_real = noise_matrix
A_indep = build_independent_noise_matrix(assignment_matrices)
compare_noise_models(A_real, A_indep)

plot_noise_matrix(A_real, config, title="Correlated 16x16", show=True)
plot_noise_matrix(A_indep, config, title="Independent Kronecker", show=True)
plot_matrix_difference(
    A_real - A_indep,
    config,
    title="Difference: Correlated - Independent",
    show=True,
)

print("Row sums (real):", np.round(A_real.sum(axis=1), 6))
print("Row sums (indep):", np.round(A_indep.sum(axis=1), 6))"""
        ),
    ]


def main() -> None:
    old = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    nb = {
        "cells": build_cells(),
        "metadata": old.get("metadata", {}),
        "nbformat": old.get("nbformat", 4),
        "nbformat_minor": old.get("nbformat_minor", 5),
    }
    NOTEBOOK_PATH.write_text(
        json.dumps(nb, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"Wrote {len(nb['cells'])} cells to {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
