import math
import random

# ── Constants ────────────────────────────────────────────────────────

HIDDEN_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 500
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8
TRAIN_RATIO = 0.8

# ── Iris Dataset (Fisher, 1936) ──────────────────────────────────────

IRIS_DATA = [
    (5.1,3.5,1.4,0.2,0),(4.9,3.0,1.4,0.2,0),(4.7,3.2,1.3,0.2,0),
    (4.6,3.1,1.5,0.2,0),(5.0,3.6,1.4,0.2,0),(5.4,3.9,1.7,0.4,0),
    (4.6,3.4,1.4,0.3,0),(5.0,3.4,1.5,0.2,0),(4.4,2.9,1.4,0.2,0),
    (4.9,3.1,1.5,0.1,0),(5.4,3.7,1.5,0.2,0),(4.8,3.4,1.6,0.2,0),
    (4.8,3.0,1.4,0.1,0),(4.3,3.0,1.1,0.1,0),(5.8,4.0,1.2,0.2,0),
    (5.7,4.4,1.5,0.4,0),(5.4,3.9,1.3,0.4,0),(5.1,3.5,1.4,0.3,0),
    (5.7,3.8,1.7,0.3,0),(5.1,3.8,1.5,0.3,0),(5.4,3.4,1.7,0.2,0),
    (5.1,3.7,1.5,0.4,0),(4.6,3.6,1.0,0.2,0),(5.1,3.3,1.7,0.5,0),
    (4.8,3.4,1.9,0.2,0),(5.0,3.0,1.6,0.2,0),(5.0,3.4,1.6,0.4,0),
    (5.2,3.5,1.5,0.2,0),(5.2,3.4,1.4,0.2,0),(4.7,3.2,1.6,0.2,0),
    (4.8,3.1,1.6,0.2,0),(5.4,3.4,1.5,0.4,0),(5.2,4.1,1.5,0.1,0),
    (5.5,4.2,1.4,0.2,0),(4.9,3.1,1.5,0.2,0),(5.0,3.2,1.2,0.2,0),
    (5.5,3.5,1.3,0.2,0),(4.9,3.6,1.4,0.1,0),(4.4,3.0,1.3,0.2,0),
    (5.1,3.4,1.5,0.2,0),(5.0,3.5,1.3,0.3,0),(4.5,2.3,1.3,0.3,0),
    (4.4,3.2,1.3,0.2,0),(5.0,3.5,1.6,0.6,0),(5.1,3.8,1.9,0.4,0),
    (4.8,3.0,1.4,0.3,0),(5.1,3.8,1.6,0.2,0),(4.6,3.2,1.4,0.2,0),
    (5.3,3.7,1.5,0.2,0),(5.0,3.3,1.4,0.2,0),
    (7.0,3.2,4.7,1.4,1),(6.4,3.2,4.5,1.5,1),(6.9,3.1,4.9,1.5,1),
    (5.5,2.3,4.0,1.3,1),(6.5,2.8,4.6,1.5,1),(5.7,2.8,4.5,1.3,1),
    (6.3,3.3,4.7,1.6,1),(4.9,2.4,3.3,1.0,1),(6.6,2.9,4.6,1.3,1),
    (5.2,2.7,3.9,1.4,1),(5.0,2.0,3.5,1.0,1),(5.9,3.0,4.2,1.5,1),
    (6.0,2.2,4.0,1.0,1),(6.1,2.9,4.7,1.4,1),(5.6,2.9,3.6,1.3,1),
    (6.7,3.1,4.4,1.4,1),(5.6,3.0,4.5,1.5,1),(5.8,2.7,4.1,1.0,1),
    (6.2,2.2,4.5,1.5,1),(5.6,2.5,3.9,1.1,1),(5.9,3.2,4.8,1.8,1),
    (6.1,2.8,4.0,1.3,1),(6.3,2.5,4.9,1.5,1),(6.1,2.8,4.7,1.2,1),
    (6.4,2.9,4.3,1.3,1),(6.6,3.0,4.4,1.4,1),(6.8,2.8,4.8,1.4,1),
    (6.7,3.0,5.0,1.7,1),(6.0,2.9,4.5,1.5,1),(5.7,2.6,3.5,1.0,1),
    (5.5,2.4,3.8,1.1,1),(5.5,2.4,3.7,1.0,1),(5.8,2.7,3.9,1.2,1),
    (6.0,2.7,5.1,1.6,1),(5.4,3.0,4.5,1.5,1),(6.0,3.4,4.5,1.6,1),
    (6.7,3.1,4.7,1.5,1),(6.3,2.3,4.4,1.3,1),(5.6,3.0,4.1,1.3,1),
    (5.5,2.5,4.0,1.3,1),(5.5,2.6,4.4,1.2,1),(6.1,3.0,4.6,1.4,1),
    (5.8,2.6,4.0,1.2,1),(5.0,2.3,3.3,1.0,1),(5.6,2.7,4.2,1.3,1),
    (5.7,3.0,4.2,1.2,1),(5.7,2.9,4.2,1.3,1),(6.2,2.9,4.3,1.3,1),
    (5.1,2.5,3.0,1.1,1),(5.7,2.8,4.1,1.3,1),
    (6.3,3.3,6.0,2.5,2),(5.8,2.7,5.1,1.9,2),(7.1,3.0,5.9,2.1,2),
    (6.3,2.9,5.6,1.8,2),(6.5,3.0,5.8,2.2,2),(7.6,3.0,6.6,2.1,2),
    (4.9,2.5,4.5,1.7,2),(7.3,2.9,6.3,1.8,2),(6.7,2.5,5.8,1.8,2),
    (7.2,3.6,6.1,2.5,2),(6.5,3.2,5.1,2.0,2),(6.4,2.7,5.3,1.9,2),
    (6.8,3.0,5.5,2.1,2),(5.7,2.5,5.0,2.0,2),(5.8,2.8,5.1,2.4,2),
    (6.4,3.2,5.3,2.3,2),(6.5,3.0,5.5,1.8,2),(7.7,3.8,6.7,2.2,2),
    (7.7,2.6,6.9,2.3,2),(6.0,2.2,5.0,1.5,2),(6.9,3.2,5.7,2.3,2),
    (5.6,2.8,4.9,2.0,2),(7.7,2.8,6.7,2.0,2),(6.3,2.7,4.9,1.8,2),
    (6.7,3.3,5.7,2.1,2),(7.2,3.2,6.0,1.8,2),(6.2,2.8,4.8,1.8,2),
    (6.1,3.0,4.9,1.8,2),(6.4,2.8,5.6,2.1,2),(7.2,3.0,5.8,1.6,2),
    (7.4,2.8,6.1,1.9,2),(7.9,3.8,6.4,2.0,2),(6.4,2.8,5.6,2.2,2),
    (6.3,2.8,5.1,1.5,2),(6.1,2.6,5.6,1.4,2),(7.7,3.0,6.1,2.3,2),
    (6.3,3.4,5.6,2.4,2),(6.4,3.1,5.5,1.8,2),(6.0,3.0,4.8,1.8,2),
    (6.9,3.1,5.4,2.1,2),(6.7,3.1,5.6,2.4,2),(6.9,3.1,5.1,2.3,2),
    (5.8,2.7,5.1,1.9,2),(6.8,3.2,5.9,2.3,2),(6.7,3.3,5.7,2.5,2),
    (6.7,3.0,5.2,2.3,2),(6.3,2.5,5.0,1.9,2),(6.5,3.0,5.2,2.0,2),
    (6.2,3.4,5.4,2.3,2),(5.9,3.0,5.1,1.8,2),
]


def load_iris():
    X = [[r[0], r[1], r[2], r[3]] for r in IRIS_DATA]
    y = [r[4] for r in IRIS_DATA]
    return X, y


def split_data(X, y, ratio, seed=42):
    rng = random.Random(seed)
    indices = list(range(len(X)))
    rng.shuffle(indices)
    n_train = int(len(X) * ratio)
    X_train = [X[i] for i in indices[:n_train]]
    y_train = [y[i] for i in indices[:n_train]]
    X_test = [X[i] for i in indices[n_train:]]
    y_test = [y[i] for i in indices[n_train:]]
    return X_train, y_train, X_test, y_test


def normalize(X_train, X_test):
    n = len(X_train)
    n_f = len(X_train[0])
    means = [sum(X_train[i][j] for i in range(n)) / n for j in range(n_f)]
    stds = [
        math.sqrt(sum((X_train[i][j] - means[j]) ** 2 for i in range(n)) / n)
        for j in range(n_f)
    ]
    stds = [s if s > 1e-12 else 1.0 for s in stds]
    X_tr = [[(x[j] - means[j]) / stds[j] for j in range(n_f)] for x in X_train]
    X_te = [[(x[j] - means[j]) / stds[j] for j in range(n_f)] for x in X_test]
    return X_tr, X_te


# ── Activation Functions ─────────────────────────────────────────────

def sigmoid(z):
    z = max(-500.0, min(500.0, z))
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1.0 - s)


def relu(z):
    return max(0.0, z)


def relu_deriv(z):
    return 1.0 if z > 0 else 0.0


def bipolar_sigmoid(z):
    return 2.0 * sigmoid(z) - 1.0


def bipolar_sigmoid_deriv(z):
    s = sigmoid(z)
    return 2.0 * s * (1.0 - s)


def gelu(z):
    return 0.5 * z * (1.0 + math.tanh(
        math.sqrt(2.0 / math.pi) * (z + 0.044715 * z ** 3)))


def gelu_deriv(z):
    c = math.sqrt(2.0 / math.pi)
    u = c * (z + 0.044715 * z ** 3)
    du = c * (1.0 + 0.134145 * z ** 2)
    tanh_u = math.tanh(u)
    return 0.5 * (1.0 + tanh_u) + 0.5 * z * (1.0 - tanh_u ** 2) * du


ACTIVATIONS = {
    'ReLU': (relu, relu_deriv),
    'Sigmoid': (sigmoid, sigmoid_deriv),
    'Bipolar Sigmoid': (bipolar_sigmoid, bipolar_sigmoid_deriv),
    'GELU': (gelu, gelu_deriv),
}


def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


# ── MLP ──────────────────────────────────────────────────────────────

def validate_params(n_input, n_hidden, n_output):
    if n_input <= 0:
        raise ValueError("Число входов должно быть положительным")
    if n_hidden <= 0:
        raise ValueError("Число нейронов скрытого слоя должно быть положительным")
    if n_output <= 0:
        raise ValueError("Число выходов должно быть положительным")


def init_weights(n_input, n_hidden, n_output, seed=42):
    validate_params(n_input, n_hidden, n_output)
    rng = random.Random(seed)
    l1 = math.sqrt(6.0 / (n_input + n_hidden))
    l2 = math.sqrt(6.0 / (n_hidden + n_output))
    W1 = [[rng.uniform(-l1, l1) for _ in range(n_input)]
           for _ in range(n_hidden)]
    b1 = [0.0] * n_hidden
    W2 = [[rng.uniform(-l2, l2) for _ in range(n_hidden)]
           for _ in range(n_output)]
    b2 = [0.0] * n_output
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


def forward(x, weights, act_fn):
    W1, b1 = weights['W1'], weights['b1']
    W2, b2 = weights['W2'], weights['b2']
    ni, nh, no = len(x), len(b1), len(b2)
    z1 = [sum(W1[i][j] * x[j] for j in range(ni)) + b1[i]
          for i in range(nh)]
    h = [act_fn(v) for v in z1]
    z2 = [sum(W2[k][i] * h[i] for i in range(nh)) + b2[k]
          for k in range(no)]
    y = softmax(z2)
    return {'z1': z1, 'h': h, 'z2': z2, 'y': y}


def cross_entropy(y_pred, label):
    return -math.log(max(y_pred[label], 1e-15))


def backward(x, label, cache, weights, act_deriv):
    W2 = weights['W2']
    z1, h, y = cache['z1'], cache['h'], cache['y']
    ni, nh, no = len(x), len(z1), len(y)
    dz2 = [y[k] - (1.0 if k == label else 0.0) for k in range(no)]
    dW2 = [[dz2[k] * h[i] for i in range(nh)] for k in range(no)]
    db2 = list(dz2)
    dh = [sum(W2[k][i] * dz2[k] for k in range(no)) for i in range(nh)]
    dz1 = [dh[i] * act_deriv(z1[i]) for i in range(nh)]
    dW1 = [[dz1[i] * x[j] for j in range(ni)] for i in range(nh)]
    db1 = list(dz1)
    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}


# ── ADAM Optimizer ───────────────────────────────────────────────────

def _zeros_mat(rows, cols):
    return [[0.0] * cols for _ in range(rows)]


def init_adam(weights):
    nh, ni = len(weights['W1']), len(weights['W1'][0])
    no = len(weights['W2'])
    return {
        'mW1': _zeros_mat(nh, ni), 'vW1': _zeros_mat(nh, ni),
        'mb1': [0.0] * nh, 'vb1': [0.0] * nh,
        'mW2': _zeros_mat(no, nh), 'vW2': _zeros_mat(no, nh),
        'mb2': [0.0] * no, 'vb2': [0.0] * no,
        't': 0,
    }


def adam_update(weights, grads, state, lr=LEARNING_RATE):
    state['t'] += 1
    t = state['t']
    bc1 = 1.0 - BETA1 ** t
    bc2 = 1.0 - BETA2 ** t
    for wk, gk, mk, vk in [('W1', 'dW1', 'mW1', 'vW1'),
                             ('W2', 'dW2', 'mW2', 'vW2')]:
        W, G, M, V = weights[wk], grads[gk], state[mk], state[vk]
        for i in range(len(W)):
            for j in range(len(W[0])):
                M[i][j] = BETA1 * M[i][j] + (1 - BETA1) * G[i][j]
                V[i][j] = BETA2 * V[i][j] + (1 - BETA2) * G[i][j] ** 2
                W[i][j] -= lr * (M[i][j] / bc1) / (math.sqrt(V[i][j] / bc2) + EPSILON)
    for wk, gk, mk, vk in [('b1', 'db1', 'mb1', 'vb1'),
                             ('b2', 'db2', 'mb2', 'vb2')]:
        b, G, M, V = weights[wk], grads[gk], state[mk], state[vk]
        for i in range(len(b)):
            M[i] = BETA1 * M[i] + (1 - BETA1) * G[i]
            V[i] = BETA2 * V[i] + (1 - BETA2) * G[i] ** 2
            b[i] -= lr * (M[i] / bc1) / (math.sqrt(V[i] / bc2) + EPSILON)


# ── Training & Evaluation ───────────────────────────────────────────

def train(X_train, y_train, n_hidden, activation_name,
          epochs=EPOCHS, lr=LEARNING_RATE, seed=42):
    act_fn, act_deriv = ACTIVATIONS[activation_name]
    n_input = len(X_train[0])
    n_output = max(y_train) + 1
    weights = init_weights(n_input, n_hidden, n_output, seed)
    state = init_adam(weights)
    rng = random.Random(seed + 1)
    for _ in range(epochs):
        idx = list(range(len(X_train)))
        rng.shuffle(idx)
        for i in idx:
            cache = forward(X_train[i], weights, act_fn)
            grads = backward(X_train[i], y_train[i], cache, weights, act_deriv)
            adam_update(weights, grads, state, lr)
    return weights


def predict(x, weights, act_fn):
    y = forward(x, weights, act_fn)['y']
    return y.index(max(y))


def accuracy(X, y, weights, act_fn):
    return sum(1 for i in range(len(X))
               if predict(X[i], weights, act_fn) == y[i]) / len(X)


def compute_loss(X, y, weights, act_fn):
    return sum(cross_entropy(forward(X[i], weights, act_fn)['y'], y[i])
               for i in range(len(X))) / len(X)


def compare_activations(n_hidden=HIDDEN_SIZE, epochs=EPOCHS,
                        lr=LEARNING_RATE, seed=42):
    X, y = load_iris()
    X_train, y_train, X_test, y_test = split_data(X, y, TRAIN_RATIO, seed)
    X_train, X_test = normalize(X_train, X_test)
    results = []
    for name in ['ReLU', 'Sigmoid', 'Bipolar Sigmoid', 'GELU']:
        act_fn = ACTIVATIONS[name][0]
        w = train(X_train, y_train, n_hidden, name, epochs, lr, seed)
        results.append({
            'name': name,
            'train_acc': accuracy(X_train, y_train, w, act_fn),
            'test_acc': accuracy(X_test, y_test, w, act_fn),
            'train_loss': compute_loss(X_train, y_train, w, act_fn),
        })
    return results


def print_results(results, n_hidden=HIDDEN_SIZE, epochs=EPOCHS,
                  lr=LEARNING_RATE):
    print(f"Датасет: Iris (150 образцов, 4 признака, 3 класса)")
    print(f"Скрытый слой: {n_hidden} нейронов")
    print(f"Оптимизатор: ADAM (lr={lr}, beta1={BETA1}, beta2={BETA2})")
    print(f"Эпохи: {epochs}")
    print()
    header = ("Сравнение функций активации скрытого слоя:")
    sep = "+-------------------+------------------+-----------------+----------------+"
    print(header)
    print(sep)
    print("| Функция активации | Точность (обуч.) | Точность (тест) | Потери (обуч.) |")
    print(sep)
    for r in results:
        n = r['name'].ljust(17)
        ta = f"{r['train_acc']*100:.2f}%".rjust(16)
        te = f"{r['test_acc']*100:.2f}%".rjust(15)
        lo = f"{r['train_loss']:.4f}".rjust(14)
        print(f"| {n} |{ta} |{te} |{lo} |")
    print(sep)


def main():
    try:
        results = compare_activations()
        print_results(results)
    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
