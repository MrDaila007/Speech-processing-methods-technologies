#!/usr/bin/env python3
"""Лабораторная работа 2: сжатие блоков изображения машинами Больцмана.

Восемь крат объясняется так, как в условии: у гауссовско-бернуллиевской машины
с одинаковым числом видимых и скрытых нейронов код бинарный. Блок 8×8 занимает
64 байта, 64 бита кода занимают 8 байт.

Запуск:
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate base
    python train.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "data"
RESULTS_PATH = ROOT / "results.json"
REPORT_PATH = ROOT / "report.tex"

EPOCHS = 8
BATCH_SIZE = 256
SEED = 42
PATCH = 8
GAUSS_LR = 1e-3
BERN_LR = 1e-2
MOMENTUM = 0.5
WEIGHT_DECAY = 2e-4

# Видимый слой всегда 64 нейрона (блок 8×8, один байт на пиксель).
# Скрытый бинарный слой из n нейронов занимает n/8 байт.
# 8×: 64 бита; 4× у первой машины: 128 бит; 2× у первой: 256 бит.
# Дальше бернуллиевские машины уменьшают уже бинарный код вдвое.
ARCHITECTURES = [
    {
        "id": "gb",
        "title": "ГБ 8$\\times$",
        "layers": [(64, "gauss")],
        "note": "одна машина Гаусс--Бернулли, одинаковое число нейронов",
    },
    {
        "id": "gb-bb",
        "title": "ГБ 4$\\times$ + ББ 2$\\times$",
        "layers": [(128, "gauss"), (64, "bern")],
        "note": "Гаусс--Бернулли с 4-кратным сжатием и Бернулли--Бернулли с 2-кратным",
    },
    {
        "id": "gb-bb-bb",
        "title": "ГБ 2$\\times$ + две ББ 2$\\times$",
        "layers": [(256, "gauss"), (128, "bern"), (64, "bern")],
        "note": "Гаусс--Бернулли с 2-кратным сжатием и две Бернулли--Бернулли с 2-кратным",
    },
]

ALGORITHMS = [
    {"id": "CD-1", "title": "CD-1", "k": 1, "persistent": False},
    {"id": "CD-10", "title": "CD-10", "k": 10, "persistent": False},
    {"id": "PCD", "title": "PCD", "k": 1, "persistent": True},
]


class RBM:
    def __init__(self, n_visible: int, n_hidden: int, kind: str, device: torch.device) -> None:
        if kind not in ("gauss", "bern"):
            raise ValueError(kind)
        self.kind = kind
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.device = device
        generator = torch.Generator(device="cpu")
        generator.manual_seed(SEED + n_visible * 17 + n_hidden)
        weight = torch.randn(n_visible, n_hidden, generator=generator) * 0.01
        self.W = weight.to(device)
        self.b = torch.zeros(n_visible, device=device)
        self.c = torch.zeros(n_hidden, device=device)
        self.vW = torch.zeros_like(self.W)
        self.vb = torch.zeros_like(self.b)
        self.vc = torch.zeros_like(self.c)
        self.persist: torch.Tensor | None = None

    def hidden_prob(self, visible: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(visible @ self.W + self.c)

    def visible_parameter(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden @ self.W.T + self.b

    def reconstruct_visible(self, hidden_binary: torch.Tensor) -> torch.Tensor:
        mean = self.visible_parameter(hidden_binary)
        if self.kind == "gauss":
            return mean
        return torch.sigmoid(mean)

    def contrastive_update(
        self,
        v0: torch.Tensor,
        k: int,
        lr: float,
        persistent: bool,
    ) -> None:
        h0_prob = self.hidden_prob(v0)
        if persistent:
            if self.persist is None or self.persist.shape != v0.shape:
                self.persist = torch.randn_like(v0) if self.kind == "gauss" else torch.bernoulli(
                    torch.full_like(v0, 0.5)
                )
            vk = self.persist
        else:
            vk = v0
        hk = torch.bernoulli(self.hidden_prob(vk))
        for _ in range(k):
            visible_mean = self.visible_parameter(hk)
            if self.kind == "gauss":
                vk = visible_mean
            else:
                vk = torch.bernoulli(torch.sigmoid(visible_mean))
            hk = torch.bernoulli(self.hidden_prob(vk))
        hk_prob = self.hidden_prob(vk)
        batch = v0.shape[0]
        dW = (v0.T @ h0_prob - vk.T @ hk_prob) / batch - WEIGHT_DECAY * self.W
        db = (v0 - vk).mean(dim=0)
        dc = (h0_prob - hk_prob).mean(dim=0)
        self.vW.mul_(MOMENTUM).add_(dW)
        self.vb.mul_(MOMENTUM).add_(db)
        self.vc.mul_(MOMENTUM).add_(dc)
        self.W.add_(self.vW, alpha=lr)
        self.b.add_(self.vb, alpha=lr)
        self.c.add_(self.vc, alpha=lr)
        if persistent:
            self.persist = vk.detach()


def cifar_patches(train: bool) -> np.ndarray:
    dataset = datasets.CIFAR10(DATA_ROOT, train=train, download=True)
    image = dataset.data.astype(np.float32) / 255.0
    gray = (
        0.2989 * image[:, :, :, 0]
        + 0.5870 * image[:, :, :, 1]
        + 0.1140 * image[:, :, :, 2]
    )
    count = gray.shape[0]
    blocks = gray.reshape(count, 4, PATCH, 4, PATCH)
    blocks = blocks.transpose(0, 1, 3, 2, 4).reshape(-1, PATCH * PATCH)
    return np.ascontiguousarray(blocks)


def compression_ratio(n_visible: int, n_code: int) -> float:
    original_bytes = float(n_visible)  # uint8
    code_bytes = n_code / 8.0
    return original_bytes / code_bytes


def train_rbm(rbm: RBM, data: torch.Tensor, k: int, persistent: bool, lr: float, label: str) -> None:
    count = data.shape[0]
    probe = data[:1024]
    for epoch in range(1, EPOCHS + 1):
        order = torch.randperm(count, device=data.device)
        for start in range(0, count - BATCH_SIZE + 1, BATCH_SIZE):
            batch = data[order[start : start + BATCH_SIZE]]
            rbm.contrastive_update(batch, k=k, lr=lr, persistent=persistent)
        with torch.no_grad():
            code = (rbm.hidden_prob(probe) >= 0.5).float()
            restored = rbm.reconstruct_visible(code)
            mse = torch.mean((restored - probe) ** 2).item()
        print(f"  {label} epoch {epoch:02d}/{EPOCHS}  probe MSE {mse:.4f}", flush=True)
        if not np.isfinite(mse):
            raise RuntimeError(f"{label}: non-finite reconstruction")


def encode_probabilities(rbm: RBM, data: torch.Tensor) -> torch.Tensor:
    parts = []
    for start in range(0, data.shape[0], 8192):
        parts.append(rbm.hidden_prob(data[start : start + 8192]))
    return torch.cat(parts, dim=0)


def fit_stack(
    layers: list[tuple[int, str]],
    train_data: torch.Tensor,
    algorithm: dict,
) -> list[RBM]:
    machines: list[RBM] = []
    visible = train_data
    for index, (n_hidden, kind) in enumerate(layers, start=1):
        machine = RBM(visible.shape[1], n_hidden, kind, train_data.device)
        lr = GAUSS_LR if kind == "gauss" else BERN_LR
        label = f"{algorithm['id']} L{index} {kind} {visible.shape[1]}->{n_hidden}"
        print(label, flush=True)
        train_rbm(machine, visible, algorithm["k"], algorithm["persistent"], lr, label)
        machines.append(machine)
        if index != len(layers):
            with torch.no_grad():
                visible = encode_probabilities(machine, visible)
    return machines


@torch.no_grad()
def decompress(machines: list[RBM], code: torch.Tensor) -> torch.Tensor:
    hidden = code
    for machine in reversed(machines[1:]):
        hidden = torch.sigmoid(hidden @ machine.W.T + machine.b)
    bottom = machines[0]
    return hidden @ bottom.W.T + bottom.b


@torch.no_grad()
def compress(machines: list[RBM], visible: torch.Tensor) -> torch.Tensor:
    value = visible
    for machine in machines:
        value = (machine.hidden_prob(value) >= 0.5).float()
    return value


@torch.no_grad()
def metrics(
    machines: list[RBM],
    test_std: torch.Tensor,
    mean: float,
    std: float,
) -> dict[str, float]:
    total_sq = 0.0
    total_pixels = 0
    psnr_sum = 0.0
    ssim_sum = 0.0
    count = 0
    c1 = (0.01) ** 2
    c2 = (0.03) ** 2
    for start in range(0, test_std.shape[0], 4096):
        batch = test_std[start : start + 4096]
        code = compress(machines, batch)
        restored = decompress(machines, code)
        original = (batch * std + mean).clamp(0.0, 1.0)
        reconstruction = (restored * std + mean).clamp(0.0, 1.0)
        diff = reconstruction - original
        total_sq += torch.sum(diff * diff).item()
        total_pixels += diff.numel()
        per_mse = torch.mean(diff * diff, dim=1).clamp_min(1e-12)
        psnr_sum += torch.sum(10.0 * torch.log10(1.0 / per_mse)).item()
        mu_x = original.mean(dim=1)
        mu_y = reconstruction.mean(dim=1)
        var_x = original.var(dim=1, unbiased=False)
        var_y = reconstruction.var(dim=1, unbiased=False)
        cov = ((original - mu_x[:, None]) * (reconstruction - mu_y[:, None])).mean(dim=1)
        numerator = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
        denominator = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
        ssim_sum += torch.sum(numerator / denominator).item()
        count += batch.shape[0]
    mse = total_sq / total_pixels
    return {"mse": mse, "psnr": psnr_sum / count, "ssim": ssim_sum / count}


def fmt_num(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}".replace(".", "{,}")


def write_report(payload: dict) -> None:
    def cell(arch_id: str, algo_id: str, key: str, digits: int) -> str:
        for run in payload["runs"]:
            if run["architecture_id"] == arch_id and run["algorithm"] == algo_id:
                return fmt_num(run[key], digits)
        raise KeyError((arch_id, algo_id, key))

    def table(key: str, digits: int, caption: str, number: int) -> str:
        lines = []
        for arch in ARCHITECTURES:
            values = " & ".join(cell(arch["id"], algo["id"], key, digits) for algo in ALGORITHMS)
            lines.append(f"{arch['title']} & {values} \\\\")
        body = "\n".join(lines)
        return rf"""\begin{{center}}
\begin{{tabular}}{{lccc}}
\toprule
Архитектура & CD-1 & CD-10 & PCD \\
\midrule
{body}
\bottomrule
\end{{tabular}}\\[0.4em]
{{\small Таблица {number}. {caption}}}
\end{{center}}"""

    best = min(payload["runs"], key=lambda run: run["mse"])
    ratio = payload["runs"][0]["compression_ratio"]
    tex = r"""\documentclass[12pt,a4paper]{article}
\usepackage{fontspec}
\usepackage[russian]{babel}
\setmainfont{Liberation Serif}
\usepackage{geometry}
\usepackage{booktabs}
\usepackage{amsmath}
\geometry{left=2.2cm,right=1.8cm,top=2cm,bottom=2cm}
\setlength{\parskip}{0.35em}
\setlength{\parindent}{1.25em}
\emergencystretch=3em

\begin{document}

\begin{center}
{\Large Лабораторная работа \textnumero~2}\\[0.4em]
{\large Глубокие доверительные нейронные сети}\\[0.6em]
1 октября 2026~г.
\end{center}

\section{Постановка задачи}

Нужно сжать изображения ограниченными машинами Больцмана, обученными с нуля, и оценить качество восстановления по MSE, PSNR и SSIM. Коэффициент сжатия равен 8: после кодирования представление занимает в восемь раз меньше байт, чем исходный блок.

Сравниваются три состава слоёв и три алгоритма обучения: CD-1, CD-10 и PCD.

\section{Выборка}

Взята CIFAR-10. Цветной кадр $32\times32$ переводится в яркость
\[
Y = 0{,}2989\,R + 0{,}5870\,G + 0{,}1140\,B
\]
и без перекрытия режется на блоки $8\times8$. Из одного изображения получается 16 блоков. Обучающая часть~--- все 50\,000 изображений (800\,000 блоков), тестовая~--- все 10\,000 изображений (160\,000 блоков).

Пиксели лежат в $[0,1]$. Перед обучением блоки стандартизуются по среднему и стандартному отклонению обучающих пикселей. Метрики считаются после обратного преобразования и отсечения в $[0,1]$, то есть в шкале исходной яркости.

\section{Сжатие и архитектуры}

Блок $8\times8$ в исходном виде занимает 64 байта. Скрытые нейроны бинарные, поэтому $n$ нейронов верхнего слоя~--- это $n$ бит, или $n/8$ байт. Если видимых и скрытых нейронов поровну ($n = 64$), код занимает 8 байт, и сжатие восьмикратное. Это как раз случай одной машины Гаусс--Бернулли из условия.

Промежуточное <<сжатие в $k$ раз>> для гауссовско-бернуллиевской машины понимается так же, через байты: при $k = 4$ скрытых нейронов 128, при $k = 2$ их 256. Следующие машины Бернулли--Бернулли работают уже с бинарным кодом, и двукратное сжатие у них означает уменьшение числа нейронов вдвое. У всех трёх итоговых схем верхний код состоит из 64 бит, измеренный коэффициент сжатия равен @@RATIO@@.

\begin{itemize}
\item Одна машина Гаусс--Бернулли, 64 видимых и 64 скрытых нейрона.
\item Гаусс--Бернулли $64 \to 128$ (4 раза по байтам) и Бернулли--Бернулли $128 \to 64$ (ещё в 2 раза).
\item Гаусс--Бернулли $64 \to 256$ (2 раза по байтам), затем Бернулли--Бернулли $256 \to 128$ и $128 \to 64$.
\end{itemize}

Дисперсия гауссовских видимых нейронов зафиксирована и равна 1, данные к этому приведены стандартизацией. Смещения и веса есть у обоих слоёв. Веса инициализируются нормальным шумом с отклонением $0{,}01$.

\section{Алгоритм обучения}

Обучение послойное и жадное. Нижняя машина видит стандартизованные пиксели. Каждая следующая получает вероятности скрытых нейронов предыдущей, а не жёсткие биты.

Контрастивная дивергенция CD-$k$ (Хинтон): положительная фаза считается по мини-выборке, скрытые вероятности не семплируются; отрицательная фаза~--- $k$ шагов выборки Гиббса, запущенных от этой же мини-выборки. В градиент отрицательной фазы входят вероятности последнего скрытого состояния. CD-1 использует один шаг, CD-10~--- десять.

PCD~--- устойчивая контрастивная дивергенция (Тилеман, 2008). Отрицательная фаза не запускается заново от данных: хранится набор частиц размера мини-выборки, и на каждом шаге он продвигается одним шагом Гиббса.

Для гауссовского видимого слоя на отрицательной фазе берётся условное среднее, а не гауссовская выборка. Скрытый слой всегда семплируется из распределения Бернулли. У бернуллиевского видимого слоя семплируются оба слоя.

Шаг обучения $10^{-3}$ для машины Гаусс--Бернулли и $10^{-2}$ для Бернулли--Бернулли, импульс $0{,}5$, затухание весов $2\cdot10^{-4}$, мини-выборка 256, 8 эпох на каждую машину.

В сам сжатый файл записывается только верхний бинарный код: единица, если вероятность скрытого нейрона не меньше $0{,}5$. Восстановление идёт сверху вниз детерминированно. Промежуточные бернуллиевские слои дают сигмоиду, нижний гауссовский слой~--- условное среднее. Затем обратная стандартизация и отсечение в $[0,1]$.

\section{Метрики}

Пусть $x$~--- исходный блок, $\hat x$~--- восстановленный, оба в $[0,1]$, $L = 1$.
\[
\mathrm{MSE} = \frac{1}{N}\sum_{i=1}^{N}(x_i-\hat x_i)^2,
\qquad
\mathrm{PSNR} = 10\log_{10}\frac{L^2}{\mathrm{MSE}_{\text{блока}}}.
\]
В таблице MSE усреднена по всем тестовым пикселям. PSNR усреднён по блокам. SSIM тоже усреднён по блокам; для блока $8\times8$ окно~--- весь блок:
\[
\mathrm{SSIM}(x,\hat x)
= \frac{(2\mu_x\mu_{\hat x}+C_1)(2\sigma_{x\hat x}+C_2)}
{(\mu_x^2+\mu_{\hat x}^2+C_1)(\sigma_x^2+\sigma_{\hat x}^2+C_2)},
\]
где $C_1 = (0{,}01)^2$ и $C_2 = (0{,}03)^2$.

Расчёт выполнен в PyTorch @@TORCH@@ на @@GPU@@.

\section{Результаты}

@@MSE@@

@@PSNR@@

@@SSIM@@

Наименьшая ошибка восстановления~--- MSE @@BEST_MSE@@ у схемы <<@@BEST_ARCH@@>> при алгоритме @@BEST_ALGO@@. Ей соответствуют PSNR @@BEST_PSNR@@~дБ и SSIM @@BEST_SSIM@@.

\section{Вывод}

@@CONCLUSION@@

\end{document}
"""
    def mse_of(arch_id: str, algo_id: str) -> float:
        for run in payload["runs"]:
            if run["architecture_id"] == arch_id and run["algorithm"] == algo_id:
                return run["mse"]
        raise KeyError((arch_id, algo_id))

    def best_of(arch_id: str) -> dict:
        subset = [run for run in payload["runs"] if run["architecture_id"] == arch_id]
        return min(subset, key=lambda run: run["mse"])

    best_gb = best_of("gb")
    best_stack2 = best_of("gb-bb")
    best_stack3 = best_of("gb-bb-bb")
    conclusion = (
        "Три требуемых состава слоёв обучены тремя алгоритмами, коэффициент сжатия во всех девяти опытах равен 8. "
        f"Меньше всего ошибается одна машина Гаусс--Бернулли: MSE {fmt_num(best_gb['mse'])} при {best_gb['algorithm']}. "
        f"На этой же машине CD-1, CD-10 и PCD дают MSE {fmt_num(mse_of('gb', 'CD-1'))}, "
        f"{fmt_num(mse_of('gb', 'CD-10'))} и {fmt_num(mse_of('gb', 'PCD'))}. "
        f"Лучшая MSE двухслойной схемы~--- {fmt_num(best_stack2['mse'])} ({best_stack2['algorithm']}), "
        f"трёхслойной~--- {fmt_num(best_stack3['mse'])} ({best_stack3['algorithm']}). "
        "При общем бюджете 64 бита дробление кода на несколько машин восстановление не улучшило: "
        "одна машина Гаусс--Бернулли остаётся точнее обоих стеков."
    )
    tex = (
        tex.replace("@@RATIO@@", fmt_num(ratio, 1))
        .replace("@@TORCH@@", payload["torch"])
        .replace("@@GPU@@", payload["gpu"])
        .replace("@@MSE@@", table("mse", 4, "MSE на тестовых блоках.", 1))
        .replace("@@PSNR@@", table("psnr", 2, "Средний PSNR тестовых блоков, дБ.", 2))
        .replace("@@SSIM@@", table("ssim", 4, "Средний SSIM тестовых блоков.", 3))
        .replace("@@BEST_MSE@@", fmt_num(best["mse"], 4))
        .replace("@@BEST_ARCH@@", best["architecture"])
        .replace("@@BEST_ALGO@@", best["algorithm"])
        .replace("@@BEST_PSNR@@", fmt_num(best["psnr"], 2))
        .replace("@@BEST_SSIM@@", fmt_num(best["ssim"], 4))
        .replace("@@CONCLUSION@@", conclusion)
    )
    REPORT_PATH.write_text(tex, encoding="utf-8")


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    train_np = cifar_patches(train=True)
    test_np = cifar_patches(train=False)
    if train_np.shape[1] != 64 or test_np.shape != (10000 * 16, 64):
        raise RuntimeError(f"unexpected patch shapes {train_np.shape} {test_np.shape}")
    # Первый блок первого изображения совпадает с левым верхним углом 8×8.
    raw = datasets.CIFAR10(DATA_ROOT, train=True, download=False).data[0].astype(np.float32) / 255.0
    gray0 = 0.2989 * raw[:, :, 0] + 0.5870 * raw[:, :, 1] + 0.1140 * raw[:, :, 2]
    if not np.allclose(train_np[0], gray0[:8, :8].reshape(-1)):
        raise RuntimeError("patch layout does not match the top-left 8x8 block")

    mean = float(train_np.mean())
    std = float(train_np.std())
    train = torch.tensor((train_np - mean) / std, device=device)
    test = torch.tensor((test_np - mean) / std, device=device)
    print(f"train patches {tuple(train.shape)}  mean {mean:.4f}  std {std:.4f}", flush=True)

    runs = []
    for arch in ARCHITECTURES:
        n_code = arch["layers"][-1][0]
        ratio = compression_ratio(64, n_code)
        if abs(ratio - 8.0) > 1e-9:
            raise RuntimeError(f"{arch['id']} compression ratio {ratio}, expected 8")
        for algorithm in ALGORITHMS:
            print(f"\n=== {arch['id']} / {algorithm['id']} ===", flush=True)
            machines = fit_stack(arch["layers"], train, algorithm)
            score = metrics(machines, test, mean, std)
            run = {
                "architecture_id": arch["id"],
                "architecture": arch["title"],
                "note": arch["note"],
                "layers": [{"hidden": n, "kind": kind} for n, kind in arch["layers"]],
                "algorithm": algorithm["id"],
                "compression_ratio": ratio,
                "code_bits": n_code,
                **score,
            }
            runs.append(run)
            print(
                f"RESULT {arch['id']} {algorithm['id']}  "
                f"MSE {score['mse']:.4f}  PSNR {score['psnr']:.2f}  SSIM {score['ssim']:.4f}",
                flush=True,
            )
            payload = {
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "mean": mean,
                "std": std,
                "train_patches": int(train.shape[0]),
                "test_patches": int(test.shape[0]),
                "runs": runs,
            }
            RESULTS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            if len(runs) == len(ARCHITECTURES) * len(ALGORITHMS):
                write_report(payload)


if __name__ == "__main__":
    main()
