import unittest
import math
from solution import (
    load_iris, split_data, normalize, sigmoid, sigmoid_deriv,
    relu, relu_deriv, bipolar_sigmoid, bipolar_sigmoid_deriv,
    gelu, gelu_deriv, softmax, validate_params, init_weights,
    forward, cross_entropy, backward, init_adam, adam_update,
    predict, accuracy, compute_loss, train, ACTIVATIONS,
)


class TestLoadIris(unittest.TestCase):
    def test_shape(self):
        X, y = load_iris()
        self.assertEqual(len(X), 150)
        self.assertEqual(len(y), 150)
        self.assertEqual(len(X[0]), 4)

    def test_labels(self):
        _, y = load_iris()
        self.assertEqual(sorted(set(y)), [0, 1, 2])
        self.assertEqual(y.count(0), 50)
        self.assertEqual(y.count(1), 50)
        self.assertEqual(y.count(2), 50)

    def test_feature_ranges(self):
        X, _ = load_iris()
        for row in X:
            for v in row:
                self.assertGreater(v, 0.0)
                self.assertLess(v, 10.0)


class TestSplitData(unittest.TestCase):
    def test_sizes(self):
        X, y = load_iris()
        Xtr, ytr, Xte, yte = split_data(X, y, 0.8, 42)
        self.assertEqual(len(Xtr), 120)
        self.assertEqual(len(ytr), 120)
        self.assertEqual(len(Xte), 30)
        self.assertEqual(len(yte), 30)

    def test_no_overlap(self):
        X, y = load_iris()
        Xtr, _, Xte, _ = split_data(X, y, 0.8, 42)
        tr_set = {tuple(x) for x in Xtr}
        te_set = {tuple(x) for x in Xte}
        self.assertEqual(len(tr_set & te_set), 0)

    def test_deterministic(self):
        X, y = load_iris()
        a1, b1, _, _ = split_data(X, y, 0.8, 42)
        a2, b2, _, _ = split_data(X, y, 0.8, 42)
        self.assertEqual(a1, a2)
        self.assertEqual(b1, b2)


class TestNormalize(unittest.TestCase):
    def test_train_mean_near_zero(self):
        X, y = load_iris()
        Xtr, _, Xte, _ = split_data(X, y, 0.8, 42)
        Xtr_n, _ = normalize(Xtr, Xte)
        n = len(Xtr_n)
        for j in range(4):
            mean = sum(Xtr_n[i][j] for i in range(n)) / n
            self.assertAlmostEqual(mean, 0.0, places=5)

    def test_train_std_near_one(self):
        X, y = load_iris()
        Xtr, _, Xte, _ = split_data(X, y, 0.8, 42)
        Xtr_n, _ = normalize(Xtr, Xte)
        n = len(Xtr_n)
        for j in range(4):
            mean = sum(Xtr_n[i][j] for i in range(n)) / n
            std = math.sqrt(sum((Xtr_n[i][j] - mean) ** 2 for i in range(n)) / n)
            self.assertAlmostEqual(std, 1.0, places=5)


class TestActivations(unittest.TestCase):
    def test_relu_positive(self):
        self.assertAlmostEqual(relu(2.0), 2.0)

    def test_relu_negative(self):
        self.assertAlmostEqual(relu(-1.0), 0.0)

    def test_relu_deriv_positive(self):
        self.assertAlmostEqual(relu_deriv(2.0), 1.0)

    def test_relu_deriv_negative(self):
        self.assertAlmostEqual(relu_deriv(-1.0), 0.0)

    def test_sigmoid_zero(self):
        self.assertAlmostEqual(sigmoid(0.0), 0.5)

    def test_sigmoid_large_positive(self):
        self.assertAlmostEqual(sigmoid(100.0), 1.0, places=5)

    def test_sigmoid_large_negative(self):
        self.assertAlmostEqual(sigmoid(-100.0), 0.0, places=5)

    def test_sigmoid_deriv_zero(self):
        self.assertAlmostEqual(sigmoid_deriv(0.0), 0.25)

    def test_bipolar_sigmoid_zero(self):
        self.assertAlmostEqual(bipolar_sigmoid(0.0), 0.0)

    def test_bipolar_sigmoid_range(self):
        self.assertAlmostEqual(bipolar_sigmoid(100.0), 1.0, places=5)
        self.assertAlmostEqual(bipolar_sigmoid(-100.0), -1.0, places=5)

    def test_bipolar_sigmoid_deriv_zero(self):
        self.assertAlmostEqual(bipolar_sigmoid_deriv(0.0), 0.5)

    def test_gelu_zero(self):
        self.assertAlmostEqual(gelu(0.0), 0.0, places=5)

    def test_gelu_positive(self):
        val = gelu(1.0)
        self.assertGreater(val, 0.8)
        self.assertLess(val, 1.0)

    def test_gelu_negative(self):
        val = gelu(-1.0)
        self.assertGreater(val, -0.2)
        self.assertLess(val, 0.0)

    def test_gelu_deriv_zero(self):
        self.assertAlmostEqual(gelu_deriv(0.0), 0.5, places=3)


class TestSoftmax(unittest.TestCase):
    def test_sums_to_one(self):
        y = softmax([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(y), 1.0)

    def test_order_preserved(self):
        y = softmax([1.0, 2.0, 3.0])
        self.assertLess(y[0], y[1])
        self.assertLess(y[1], y[2])

    def test_equal_inputs(self):
        y = softmax([1.0, 1.0, 1.0])
        for v in y:
            self.assertAlmostEqual(v, 1.0 / 3.0, places=5)

    def test_numerical_stability(self):
        y = softmax([1000.0, 1001.0, 1002.0])
        self.assertAlmostEqual(sum(y), 1.0)


class TestValidateParams(unittest.TestCase):
    def test_valid(self):
        validate_params(4, 16, 3)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            validate_params(0, 16, 3)

    def test_invalid_hidden(self):
        with self.assertRaises(ValueError):
            validate_params(4, 0, 3)

    def test_invalid_output(self):
        with self.assertRaises(ValueError):
            validate_params(4, 16, 0)


class TestInitWeights(unittest.TestCase):
    def test_dimensions(self):
        w = init_weights(4, 16, 3, seed=42)
        self.assertEqual(len(w['W1']), 16)
        self.assertEqual(len(w['W1'][0]), 4)
        self.assertEqual(len(w['b1']), 16)
        self.assertEqual(len(w['W2']), 3)
        self.assertEqual(len(w['W2'][0]), 16)
        self.assertEqual(len(w['b2']), 3)

    def test_biases_zero(self):
        w = init_weights(4, 16, 3, seed=42)
        self.assertEqual(w['b1'], [0.0] * 16)
        self.assertEqual(w['b2'], [0.0] * 3)

    def test_xavier_range(self):
        w = init_weights(4, 16, 3, seed=42)
        limit = math.sqrt(6.0 / (4 + 16))
        for row in w['W1']:
            for v in row:
                self.assertGreaterEqual(v, -limit)
                self.assertLessEqual(v, limit)

    def test_deterministic(self):
        w1 = init_weights(4, 16, 3, seed=42)
        w2 = init_weights(4, 16, 3, seed=42)
        self.assertEqual(w1['W1'], w2['W1'])


class TestForward(unittest.TestCase):
    def test_output_shape(self):
        w = init_weights(4, 16, 3, seed=42)
        c = forward([1.0, 2.0, 3.0, 4.0], w, relu)
        self.assertEqual(len(c['z1']), 16)
        self.assertEqual(len(c['h']), 16)
        self.assertEqual(len(c['z2']), 3)
        self.assertEqual(len(c['y']), 3)

    def test_softmax_output(self):
        w = init_weights(4, 16, 3, seed=42)
        c = forward([1.0, 2.0, 3.0, 4.0], w, sigmoid)
        self.assertAlmostEqual(sum(c['y']), 1.0)

    def test_probabilities_positive(self):
        w = init_weights(4, 16, 3, seed=42)
        c = forward([1.0, 2.0, 3.0, 4.0], w, relu)
        for v in c['y']:
            self.assertGreaterEqual(v, 0.0)


class TestCrossEntropy(unittest.TestCase):
    def test_perfect_prediction(self):
        loss = cross_entropy([1.0, 0.0, 0.0], 0)
        self.assertAlmostEqual(loss, 0.0, places=5)

    def test_wrong_prediction(self):
        loss = cross_entropy([0.01, 0.01, 0.98], 0)
        self.assertGreater(loss, 3.0)

    def test_uniform(self):
        loss = cross_entropy([1.0 / 3, 1.0 / 3, 1.0 / 3], 0)
        self.assertAlmostEqual(loss, math.log(3), places=5)


class TestBackward(unittest.TestCase):
    def test_gradient_shapes(self):
        w = init_weights(4, 16, 3, seed=42)
        c = forward([1.0, 2.0, 3.0, 4.0], w, relu)
        g = backward([1.0, 2.0, 3.0, 4.0], 0, c, w, relu_deriv)
        self.assertEqual(len(g['dW1']), 16)
        self.assertEqual(len(g['dW1'][0]), 4)
        self.assertEqual(len(g['db1']), 16)
        self.assertEqual(len(g['dW2']), 3)
        self.assertEqual(len(g['dW2'][0]), 16)
        self.assertEqual(len(g['db2']), 3)

    def test_output_gradient_sum(self):
        w = init_weights(4, 16, 3, seed=42)
        c = forward([1.0, 2.0, 3.0, 4.0], w, relu)
        g = backward([1.0, 2.0, 3.0, 4.0], 0, c, w, relu_deriv)
        dz2_sum = sum(g['db2'])
        self.assertAlmostEqual(dz2_sum, 0.0, places=5)


class TestAdam(unittest.TestCase):
    def test_weights_change(self):
        w = init_weights(4, 16, 3, seed=42)
        w1_before = [row[:] for row in w['W1']]
        c = forward([1.0, 2.0, 3.0, 4.0], w, relu)
        g = backward([1.0, 2.0, 3.0, 4.0], 0, c, w, relu_deriv)
        s = init_adam(w)
        adam_update(w, g, s)
        changed = any(w['W1'][i][j] != w1_before[i][j]
                      for i in range(16) for j in range(4))
        self.assertTrue(changed)

    def test_timestep_increments(self):
        w = init_weights(4, 16, 3, seed=42)
        c = forward([1.0, 2.0, 3.0, 4.0], w, relu)
        g = backward([1.0, 2.0, 3.0, 4.0], 0, c, w, relu_deriv)
        s = init_adam(w)
        adam_update(w, g, s)
        self.assertEqual(s['t'], 1)
        adam_update(w, g, s)
        self.assertEqual(s['t'], 2)


class TestTrainAndPredict(unittest.TestCase):
    def test_predict_returns_valid_class(self):
        w = init_weights(4, 16, 3, seed=42)
        p = predict([1.0, 2.0, 3.0, 4.0], w, relu)
        self.assertIn(p, [0, 1, 2])

    def test_accuracy_range(self):
        X, y = load_iris()
        Xtr, ytr, Xte, yte = split_data(X, y, 0.8, 42)
        Xtr, Xte = normalize(Xtr, Xte)
        w = train(Xtr, ytr, 16, 'ReLU', epochs=50, lr=0.001, seed=42)
        acc = accuracy(Xte, yte, w, relu)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_loss_decreases(self):
        X, y = load_iris()
        Xtr, ytr, Xte, yte = split_data(X, y, 0.8, 42)
        Xtr, Xte = normalize(Xtr, Xte)
        w_few = train(Xtr, ytr, 16, 'Sigmoid', epochs=10, lr=0.001, seed=42)
        w_more = train(Xtr, ytr, 16, 'Sigmoid', epochs=100, lr=0.001, seed=42)
        l_few = compute_loss(Xtr, ytr, w_few, sigmoid)
        l_more = compute_loss(Xtr, ytr, w_more, sigmoid)
        self.assertLess(l_more, l_few)

    def test_all_activations_learn(self):
        X, y = load_iris()
        Xtr, ytr, Xte, yte = split_data(X, y, 0.8, 42)
        Xtr, Xte = normalize(Xtr, Xte)
        for name in ACTIVATIONS:
            act_fn = ACTIVATIONS[name][0]
            w = train(Xtr, ytr, 16, name, epochs=100, lr=0.001, seed=42)
            acc = accuracy(Xtr, ytr, w, act_fn)
            self.assertGreater(acc, 0.5,
                               f"{name} train accuracy too low: {acc}")


if __name__ == "__main__":
    unittest.main()
