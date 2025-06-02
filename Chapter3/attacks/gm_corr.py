import os
import tensorflow as tf
import time
from attacks.attack import Attack
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
tf.random.set_seed(0)

# Smooth Gradient Method (SGM)


class GM_corr(Attack):
    def __init__(
        self,
        model_dir,
        x_test,
        y_test,
        out_dir,
        eps_init=0.001,
        nb_iter=1000,
        smoothness=0.01,
        mu=1.0,
        eps=0.1,
        b=250,
        beta=0.05,
        c = 0.2, 
        k = -5
    ):
        super().__init__()
        self.f = self._model_as_tf_function(tf.keras.models.load_model(model_dir))
        self.x_test = x_test
        self.y_test = y_test  # used for evaluation
        self.eps_init = eps_init
        self.nb_iter = nb_iter
        self.out_dir = out_dir
        self.b = b
        self.mu = mu  # coefficient entropy
        self.smoothness = smoothness
        self.eps = eps  # max allowed perturbation
        self.beta = beta
        self.p_x_test = None
        self.p_y_pred = None
        self.coef_shifted = None
        self.y_pred = None
        self.c = c
        self.k = k

    def cross_correlation_integration(self, input_array, x_r):
        # 使用频域来计算交叉相关
        def cross_correlation_fft(input_array, x_r):
            # 翻转操作
            flipped_x_r = tf.reverse(x_r, axis=[1])

            # 使用快速傅里叶变换计算交叉相关
            input_fft = tf.signal.fft(tf.cast(input_array, tf.complex64))
            flipped_fft = tf.signal.fft(tf.cast(flipped_x_r, tf.complex64))

            # 计算点对点乘积并应用逆FFT
            cross_corr_complex = tf.signal.ifft(input_fft * tf.math.conj(flipped_fft))
            cross_corr = tf.cast(tf.math.real(cross_corr_complex), tf.float32)

            return cross_corr

        cross_corr = cross_correlation_fft(input_array, x_r)

        # Parameters for the custom weight function


        # Calculate the weights using the custom function
        seq_length = input_array.shape[1]
        delay_values = tf.range(seq_length, dtype=tf.float32)
        normalized_delay_values = delay_values / tf.cast(
            seq_length - 1, tf.float32
        )  # 归一化到[0,1]
        weights = 1 / (1 + tf.exp(self.k * (normalized_delay_values - self.c)))
        weights = tf.expand_dims(
            weights, axis=-1
        )  # Expand dims for broadcasting on the 2nd dimension

        # Multiply with weights
        weighted_cross_corr = cross_corr * weights

        # Integrate the weighted cross-correlation result
        integrated_result = tf.reduce_sum(weighted_cross_corr, axis=1)

        return integrated_result

    def _loss_function(self, x, r, cur_x_test, y_target):
        x_r = x + r
        ce_loss = tf.keras.backend.categorical_crossentropy(y_target, self.f(x_r))

        crosscorr_loss = self.cross_correlation_integration(cur_x_test, x_r)

        # Normalize the cross-correlation loss
        # norm_x_r = tf.cast(tf.norm(x_r, axis=1), tf.float32)
        # norm_x_test = tf.cast(tf.norm(cur_x_test, axis=1), tf.float32)
        normalized_crosscorr_loss = crosscorr_loss  # / (norm_x_test * norm_x_r + 1e-7)

        # print(ce_loss)
        # print(normalized_crosscorr_loss)
        # raise KeyboardInterrupt
        return tf.reduce_mean(ce_loss + normalized_crosscorr_loss)

    def perturb(self):
        start_time = time.time()

        opt = tf.keras.optimizers.legacy.Adam()

        losses = np.zeros(shape=(self.nb_iter, 1))

        n = self.x_test.shape[0]
        self.p_x_test = self.x_test.copy()
        self.p_y_pred = np.zeros(shape=self.y_test.shape, dtype=np.float32)
        self.y_pred = np.zeros(shape=self.y_test.shape, dtype=np.float32)

        denom = 0
        # loop through batches for OOM issues
        for i in range(self.batch_size, n + self.batch_size, self.batch_size):
            denom += 1
            max_i = min(n, i)
            cur_i = i - self.batch_size

            cur_x_test = self.x_test[cur_i:max_i]
            x = tf.constant(cur_x_test, dtype=tf.float32)

            self.coef_shifted = np.ones(cur_x_test.shape)
            self.coef_shifted[:, -1, :] = 0
            self.coef_shifted = tf.constant(self.coef_shifted, dtype=tf.float32)

            # define constraint
            def constraint_eps(t):
                # t is the tensor
                return tf.clip_by_value(t, -self.eps, self.eps)

            # pick random initial perturbation
            r_data = np.random.randint(low=-1, high=1, size=x.shape) * self.eps_init
            r = tf.Variable(r_data, dtype=tf.float32, constraint=constraint_eps)

            y_target, y_pred = self._get_y_target(x)

            loss_func = lambda: self._loss_function(x, r, cur_x_test, y_target)

            for itr in range(self.nb_iter):
                opt.minimize(loss_func, var_list=[r])
                losses[itr] = losses[itr] + self._loss_function(
                    x, r, cur_x_test, y_target
                )

            self.p_x_test[cur_i:max_i] = self.p_x_test[cur_i:max_i] + r.numpy()
            self.p_y_pred[cur_i:max_i] = self.f(
                tf.constant(self.p_x_test[cur_i:max_i], dtype=tf.float32)
            ).numpy()
            self.y_pred[cur_i:max_i] = y_pred

        losses = losses / denom
        self.save_loss(losses)

        duration = time.time() - start_time

        df_metrics = self.compute_df_metrics(duration)

        df_metrics.to_csv(os.path.join(self.out_dir, "df_metrics.csv"))

        self.plot()

        tf.keras.backend.clear_session()

        return df_metrics

    def perturb_demo(self):
        start_time = time.time()

        opt = tf.keras.optimizers.legacy.Adam()

        losses = np.zeros(shape=(self.nb_iter, 1))

        n = self.x_test.shape[0]
        self.p_x_test = self.x_test.copy()
        self.p_y_pred = np.zeros(shape=self.y_test.shape, dtype=float)
        self.y_pred = np.zeros(shape=self.y_test.shape, dtype=float)

        denom = 0

        # 用于储存
        px_test = self.x_test.copy()
        py_test = self.p_y_pred.copy()

        # loop through batches for OOM issues
        for i in range(self.batch_size, n + self.batch_size, self.batch_size):
            denom += 1
            max_i = min(n, i)
            cur_i = i - self.batch_size

            cur_x_test = self.x_test[cur_i:max_i]
            x = tf.constant(cur_x_test, dtype=tf.float32)

            self.coef_shifted = np.ones(cur_x_test.shape)
            self.coef_shifted[:, -1, :] = 0
            self.coef_shifted = tf.constant(self.coef_shifted, dtype=tf.float32)

            # define constraint
            def constraint_eps(t):
                # t is the tensor
                return tf.clip_by_value(t, -self.eps, self.eps)

            # pick random initial perturbation
            r_data = np.random.randint(low=-1, high=1, size=x.shape) * self.eps_init
            r = tf.Variable(r_data, dtype=tf.float32, constraint=constraint_eps)

            y_target, y_pred = self._get_y_target(x)

            loss_func = lambda: self._loss_function(x, r, cur_x_test, y_target)

            for itr in range(self.nb_iter):
                opt.minimize(loss_func, var_list=[r])
                losses[itr] = losses[itr] + self._loss_function(
                    x, r, cur_x_test, y_target
                )

                px_test[cur_i:max_i] = self.x_test[cur_i:max_i].copy() + r.numpy()
                py_test[cur_i:max_i] = self.f(
                    tf.constant(px_test[cur_i:max_i], dtype=tf.float32)
                ).numpy()

            self.p_x_test[cur_i:max_i] = self.p_x_test[cur_i:max_i] + r.numpy()
            self.p_y_pred[cur_i:max_i] = self.f(
                tf.constant(self.p_x_test[cur_i:max_i], dtype=tf.float32)
            ).numpy()
            self.y_pred[cur_i:max_i] = y_pred

        return self.p_x_test, self.p_y_pred, self.y_pred