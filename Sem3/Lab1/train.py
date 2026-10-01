#!/usr/bin/env python3
"""Лабораторная работа 1: пять свёрточных сетей на CIFAR-10.

Запуск:
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate base
    python train.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "data"
RESULTS_PATH = ROOT / "results.json"
REPORT_PATH = ROOT / "report.tex"

EPOCHS = 20
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LeNet(nn.Module):
    """LeNet-подобная сеть для входа 32×32×3. После свёрток карта 5×5."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class CNN3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class CNNBatchNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class VGGMini(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + residual)


class ResCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.block1 = ResidualBlock(32)
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block2 = ResidualBlock(64)
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.block3 = ResidualBlock(128)
        self.fc = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.down1(x)
        x = self.block2(x)
        x = self.down2(x)
        x = self.block3(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


MODELS: list[tuple[str, str, type[nn.Module]]] = [
    ("lenet", "LeNet-подобная", LeNet),
    ("cnn3", "CNN-3", CNN3),
    ("cnn-bn", "CNN-BN", CNNBatchNorm),
    ("vgg-mini", "VGG-mini", VGGMini),
    ("rescnn", "ResCNN", ResCNN),
]

ARCHITECTURE_TEX = r"""
\begin{itemize}
\item \textbf{LeNet-подобная.} Две свёртки $5\times5$ (6 и 16 каналов) с ReLU и подвыборкой $2\times2$ после каждой. Полносвязные слои $400 \to 120 \to 84 \to 10$. От классической LeNet-5 отличается функцией ReLU и трёхканальным входом.
\item \textbf{CNN-3.} Три блока <<свёртка $3\times3$ с дополнением, ReLU, подвыборка $2\times2$>> с 32, 64 и 128 каналами. Затем полносвязные слои $2048 \to 256 \to 10$.
\item \textbf{CNN-BN.} Та же схема, что у CNN-3, но после каждой свёртки стоит пакетная нормализация, смещение у свёртки отключено.
\item \textbf{VGG-mini.} Три стадии по две свёртки $3\times3$: 32, 32, подвыборка; 64, 64, подвыборка; 128, 128, подвыборка. Классификатор $2048 \to 256 \to 10$.
\item \textbf{ResCNN.} Ствол $3\to32$, остаточный блок на 32 каналах, свёртка со шагом 2 до 64 каналов, остаточный блок, свёртка со шагом 2 до 128 каналов, остаточный блок, глобальное усреднение и линейный слой $128 \to 10$. В остаточном блоке два свёрточных слоя $3\times3$ с пакетной нормализацией, вход блока прибавляется к выходу.
\end{itemize}
"""


def fmt_num(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", "{,}")


def fmt_int(value: int) -> str:
    return f"{value:,}".replace(",", r"\,")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        predicted = model(images).argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def train_one(
    name: str,
    title: str,
    factory: type[nn.Module],
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> dict:
    set_seed(SEED)
    model = factory().to(device)
    dummy = torch.zeros(2, 3, 32, 32, device=device)
    output = model(dummy)
    if output.shape != (2, 10):
        raise RuntimeError(f"{name}: unexpected output shape {tuple(output.shape)}")
    n_params = count_parameters(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    history = []
    print(f"\n=== {title} | parameters {n_params} ===", flush=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * labels.size(0)
            seen += labels.size(0)
        train_loss = running / seen
        test_acc = evaluate(model, test_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "test_accuracy": test_acc})
        print(
            f"{title} epoch {epoch:02d}/{EPOCHS}  loss {train_loss:.4f}  test {test_acc:.2f}%",
            flush=True,
        )
    return {
        "id": name,
        "title": title,
        "parameters": n_params,
        "train_loss": history[-1]["train_loss"],
        "test_accuracy": history[-1]["test_accuracy"],
        "history": history,
    }


def write_report(payload: dict) -> None:
    rows = []
    for item in payload["models"]:
        rows.append(
            f"{item['title']} & {fmt_int(item['parameters'])} & "
            f"{fmt_num(item['train_loss'], 4)} & {fmt_num(item['test_accuracy'])} \\\\"
        )
    best = max(payload["models"], key=lambda item: item["test_accuracy"])
    worst = min(payload["models"], key=lambda item: item["test_accuracy"])
    ordered = sorted(payload["models"], key=lambda item: item["test_accuracy"])
    n_above = sum(item["test_accuracy"] > 60.0 for item in payload["models"])
    n_models = len(payload["models"])
    if n_above == n_models:
        threshold = f"Порог 60\\% превышен у всех {n_models} сетей."
    else:
        threshold = f"Порог 60\\% превышен у {n_above} из {n_models} сетей."
    listing = ", ".join(
        f"{item['title']}~--- {fmt_num(item['test_accuracy'])}\\%" for item in ordered
    )
    if n_models >= 5 and n_above >= 1:
        grade = "Условие на 10 баллов выполнено: архитектур не меньше пяти, и хотя бы одна выше 60\\%."
        if n_above == n_models:
            grade = "Условие на 10 баллов выполнено: архитектур пять, и каждая выше 60\\%."
    else:
        grade = "Порог 60\\% хотя бы для одной сети не достигнут."
    conclusion = (
        "При одном и том же алгоритме точность определяется архитектурой. По возрастанию: "
        + listing
        + ". "
        + grade
    )
    tex = r"""\documentclass[12pt,a4paper]{article}
\usepackage{fontspec}
\usepackage[russian]{babel}
\setmainfont{Liberation Serif}
\usepackage{geometry}
\usepackage{booktabs}
\usepackage{amsmath}
\geometry{left=2.5cm,right=2cm,top=2cm,bottom=2cm}
\setlength{\parskip}{0.4em}
\setlength{\parindent}{1.25em}
\emergencystretch=3em

\begin{document}

\begin{center}
{\Large Лабораторная работа \textnumero~1}\\[0.4em]
{\large Глубокие свёрточные нейронные сети}\\[0.6em]
1 октября 2026~г.
\end{center}

\section{Постановка задачи}

Нужно взять выборку CIFAR-10, собрать свёрточную сеть, обучить её с нуля и посчитать точность классификации на тестовой выборке. В работе рассмотрено пять архитектур. Для максимальной оценки достаточно, чтобы хотя бы одна из них превысила 60\% на тесте; ограничения снизу на точность остальных сетей нет.

\section{Выборка}

CIFAR-10 состоит из 60\,000 цветных изображений размером $32\times32$ пикселя, по 6\,000 на каждый из 10 классов: самолёт, автомобиль, птица, кошка, олень, собака, лягушка, лошадь, корабль, грузовик. Официальное разбиение~--- 50\,000 обучающих изображений и 10\,000 тестовых.

Пиксели переводятся в диапазон $[0,1]$ и нормируются средним и стандартным отклонением по каналам:
\[
\mu = (0{,}4914;\ 0{,}4822;\ 0{,}4465), \qquad
\sigma = (0{,}2470;\ 0{,}2435;\ 0{,}2616).
\]
На обучении к изображению применяются случайное отражение по горизонтали и случайный сдвиг: из кадра с полем ширины 4 пикселя вырезается окно $32\times32$. Тестовые изображения только нормируются.

\section{Алгоритм обучения}

Все пять сетей обучаются одним алгоритмом, с нуля, без предобученных весов.

Функция потерь~--- перекрёстная энтропия между выходом сети и меткой класса. Параметры обновляет Adam: шаг обучения $10^{-3}$, моменты $\beta_1 = 0{,}9$ и $\beta_2 = 0{,}999$, $L_2$-штраф на веса $10^{-4}$. Размер мини-выборки~--- 128 изображений, число эпох~--- 20. Начальные веса свёрток задаются схемой Кайминга, как это делает PyTorch по умолчанию.

Расчёт выполнен в PyTorch @@TORCH@@ на графическом процессоре @@GPU@@.

\section{Архитектуры}
@@ARCH@@

\section{Результаты}

Точность ниже считается на всех 10\,000 тестовых изображениях после последней, двадцатой эпохи. Это точность финальных весов, а не лучшей промежуточной точки.

\begin{center}
\begin{tabular}{lrrr}
\toprule
Архитектура & Параметры & Потеря обучения & Точность теста, \% \\
\midrule
@@ROWS@@
\bottomrule
\end{tabular}\\[0.4em]
{\small Таблица 1. Результаты пяти свёрточных сетей на CIFAR-10.}
\end{center}

Наибольшая точность у сети <<@@BEST@@>>: @@BEST_ACC@@\%. Наименьшая~--- у сети <<@@WORST@@>>: @@WORST_ACC@@\%. @@THRESHOLD@@

\section{Вывод}

@@CONCLUSION@@

\end{document}
"""
    tex = (
        tex.replace("@@TORCH@@", payload["torch"])
        .replace("@@GPU@@", payload["gpu"])
        .replace("@@ARCH@@", ARCHITECTURE_TEX.strip())
        .replace("@@ROWS@@", "\n".join(rows))
        .replace("@@BEST@@", best["title"])
        .replace("@@BEST_ACC@@", fmt_num(best["test_accuracy"]))
        .replace("@@WORST@@", worst["title"])
        .replace("@@WORST_ACC@@", fmt_num(worst["test_accuracy"]))
        .replace("@@THRESHOLD@@", threshold)
        .replace("@@CONCLUSION@@", conclusion)
    )
    REPORT_PATH.write_text(tex, encoding="utf-8")


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"device: {device}", flush=True)

    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    train_set = datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
    )

    models = []
    for name, title, factory in MODELS:
        models.append(train_one(name, title, factory, train_loader, test_loader, device))
        payload = {
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": "Adam",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "models": models,
        }
        RESULTS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        write_report(payload)
        print(f"saved {RESULTS_PATH.name} and {REPORT_PATH.name}", flush=True)


if __name__ == "__main__":
    main()
