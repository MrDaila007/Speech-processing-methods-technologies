import random
import math

FREQ_MIN = 20.0
FREQ_MAX = 20000.0
VALID_RATES = (48, 96, 128, 176, 256)


def validate_params(n_seconds, bitrate, sample_rate_khz):
    if n_seconds <= 0:
        raise ValueError("n должно быть > 0")
    if bitrate <= 0 or bitrate > 24:
        raise ValueError("Битрейт должен быть от 1 до 24")
    if sample_rate_khz not in VALID_RATES:
        raise ValueError("Недопустимая частота записи")
    return n_seconds, bitrate, sample_rate_khz


def generate_signal(count, seed=42):
    rng = random.Random(seed)
    return [rng.uniform(FREQ_MIN, FREQ_MAX) for _ in range(count)]


def compute_step(bitrate):
    levels = (1 << bitrate) - 1
    return (FREQ_MAX - FREQ_MIN) / levels


def quantize(value, step):
    relative = value - FREQ_MIN
    level = round(relative / step)
    return FREQ_MIN + level * step


def discretize(signal, n_seconds, bitrate, sample_rate_khz):
    total_input = len(signal)
    total_samples = sample_rate_khz * 1000 * n_seconds
    step = compute_step(bitrate)
    samples_per_bin = total_input / total_samples

    output = []
    for i in range(total_samples):
        start = int(i * samples_per_bin)
        end = int((i + 1) * samples_per_bin)
        if end > total_input:
            end = total_input
        if start >= end:
            start = end - 1

        avg = sum(signal[start:end]) / (end - start)
        output.append(quantize(avg, step))

    return output


def print_results(output, bitrate, sample_rate_khz, n_seconds):
    levels = (1 << bitrate) - 1
    step = compute_step(bitrate)
    total_samples = sample_rate_khz * 1000 * n_seconds

    print(f"\n=== Результаты дискретизации ===")
    print(f"Частота записи:   {sample_rate_khz} KHz")
    print(f"Битрейт:          {bitrate} бит")
    print(f"Уровней квант.:   {levels + 1}")
    print(f"Шаг квантования:  {step:.4f} Hz")
    print(f"Всего сэмплов:    {total_samples}")

    show = min(len(output), 20)
    print(f"\nПервые {show} сэмплов:")
    print(f"{'Сэмпл':>10}{'Частота (Hz)':>16}")
    print("-" * 26)
    for i in range(show):
        print(f"{i:>10}{output[i]:>16.4f}")

    out_min = min(output)
    out_max = max(output)
    out_avg = sum(output) / len(output)
    print(f"\nСтатистика:")
    print(f"  Min: {out_min:.4f} Hz")
    print(f"  Max: {out_max:.4f} Hz")
    print(f"  Avg: {out_avg:.4f} Hz")


def main():
    n_seconds = int(input("Количество секунд аудио (n): "))
    bitrate = int(input("Битрейт (бит): "))
    sample_rate_khz = int(input("Частота записи (48, 96, 128, 176, 256) KHz: "))

    n_seconds, bitrate, sample_rate_khz = validate_params(
        n_seconds, bitrate, sample_rate_khz
    )

    total_values = 1_000_000 * n_seconds
    print(f"\nГенерация {total_values} значений сигнала...")
    signal = generate_signal(total_values)

    print("Дискретизация...")
    output = discretize(signal, n_seconds, bitrate, sample_rate_khz)

    print_results(output, bitrate, sample_rate_khz, n_seconds)


if __name__ == "__main__":
    main()
