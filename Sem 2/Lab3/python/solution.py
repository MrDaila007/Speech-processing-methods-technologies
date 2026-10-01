import random
import math

N_OBSERVATIONS = 27


def validate_params(n_states):
    if n_states <= 0:
        raise ValueError("Число состояний должно быть > 0")


def char_to_index(c):
    if 'a' <= c <= 'z':
        return ord(c) - ord('a')
    if c == ' ':
        return 26
    raise ValueError("Недопустимый символ: допустимы только a-z и пробел")


def text_to_indices(text):
    if not text:
        raise ValueError("Текст не должен быть пустым")
    return [char_to_index(c) for c in text]


def normalize(values):
    total = sum(values)
    if total <= 0.0:
        raise ValueError("Невозможно нормализовать: сумма <= 0")
    return [v / total for v in values]


def init_hmm(n_states, seed=42):
    validate_params(n_states)
    rng = random.Random(seed)

    pi = normalize([rng.random() for _ in range(n_states)])

    A = []
    for _ in range(n_states):
        A.append(normalize([rng.random() for _ in range(n_states)]))

    B = []
    for _ in range(n_states):
        B.append(normalize([rng.random() for _ in range(N_OBSERVATIONS)]))

    return {
        'n_states': n_states,
        'pi': pi,
        'A': A,
        'B': B,
    }


def forward(observations, hmm):
    if not observations:
        raise ValueError("Последовательность наблюдений не должна быть пустой")
    for obs in observations:
        if obs < 0 or obs >= N_OBSERVATIONS:
            raise ValueError("Индекс наблюдения вне допустимого диапазона")

    T = len(observations)
    N = hmm['n_states']
    pi = hmm['pi']
    A = hmm['A']
    B = hmm['B']

    # Scaled forward algorithm (Rabiner)
    alpha = [[0.0] * N for _ in range(T)]
    scaling = [0.0] * T

    # Initialization
    scale = 0.0
    for i in range(N):
        alpha[0][i] = pi[i] * B[i][observations[0]]
        scale += alpha[0][i]
    if scale <= 0.0:
        return {'log_probability': float('-inf')}
    scaling[0] = 1.0 / scale
    for i in range(N):
        alpha[0][i] *= scaling[0]

    # Induction
    for t in range(1, T):
        scale = 0.0
        for j in range(N):
            s = sum(alpha[t - 1][i] * A[i][j] for i in range(N))
            alpha[t][j] = s * B[j][observations[t]]
            scale += alpha[t][j]
        if scale <= 0.0:
            return {'log_probability': float('-inf')}
        scaling[t] = 1.0 / scale
        for j in range(N):
            alpha[t][j] *= scaling[t]

    # Termination
    log_prob = -sum(math.log(c) for c in scaling)
    return {'log_probability': log_prob}


def print_results(hmm, phrases, results):
    print(f"\n=== Результаты СММ ===")
    print(f"Скрытых состояний:  {hmm['n_states']}")
    print(f"Символов наблюдения: {N_OBSERVATIONS} (a-z + пробел)")

    print(f"\nВероятности фраз:")
    print(f"{'Фраза':>30}{'Длина':>10}{'Лог-вероятность':>20}")
    print("-" * 60)
    for phrase, result in zip(phrases, results):
        display = f'"{phrase}"'
        print(f"{display:>30}{len(phrase):>10}{result['log_probability']:>20.4f}")


def main():
    n_states = int(input("Число скрытых состояний: "))

    print("Введите фразы (пустая строка для завершения):")
    phrases = []
    while True:
        line = input()
        if not line:
            break
        phrases.append(line)

    if not phrases:
        print("Ошибка: не введено ни одной фразы")
        return

    print("\nИнициализация СММ...")
    hmm = init_hmm(n_states)

    print("Вычисление вероятностей...")
    results = []
    for phrase in phrases:
        obs = text_to_indices(phrase)
        results.append(forward(obs, hmm))

    print_results(hmm, phrases, results)


if __name__ == "__main__":
    main()
