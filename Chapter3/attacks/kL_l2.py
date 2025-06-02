import os
import tensorflow as tf
import time
from attacks.attack import Attack
import numpy as np
import matplotlib.pyplot as plt


# Gradient Method (GM)

class KL_l2(Attack):
    def __init__(self, model_dir, x_test, y_test, out_dir,
                 eps_init=0.001, nb_iter=1000, smoothness=0.1, mu=1.0, eps=0.1,
                 alpha = 0.01,lamb = 1, gamma = 0.48):
        super().__init__()
        self.f = self._model_as_tf_function(tf.keras.models.load_model(model_dir))
        self.x_test = x_test
        self.y_test = y_test  # used for evaluation
        self.eps_init = eps_init
        self.nb_iter = nb_iter
        self.out_dir = out_dir
        self.eps = eps  # max allowed perturbation
   
        self.lamb = lamb
        self.gamma = gamma 
        self.alpha = alpha

        self.p_x_test = None
        self.p_y_pred = None
        self.y_pred = None

    def _get_y_target(self, x, method='average_case'):
        """
        :param x: the input tensor
        :param method: how you want to generate the target
        :return: the target predictions, the predictions for the unperturbed data
        """
        if method == 'average_case':
            y_pred = self.f(x)
            y_pred = tf.stop_gradient(y_pred).numpy()

            target_p_logits = y_pred.copy()

            for i in range(len(y_pred)):
                # 获取前两个最高的类别标签
                c_top2 = np.argsort(y_pred[i])[-2:][::-1]

                # 获取最高和次高类别的logits
                logits_top1, logits_top2 = target_p_logits[i, c_top2[0]], target_p_logits[i, c_top2[1]]

                # 计算这两个logits的平均值，并分别添加和减去0.01
                sum_ = logits_top1 + logits_top2
                target_p_logits[i, c_top2[0]], target_p_logits[i, c_top2[1]] = sum_* self.gamma , sum_* (1-self.gamma)


            target_p_logits = tf.constant(target_p_logits, dtype=tf.float32)
            return target_p_logits, y_pred
        else:
            raise Exception('Chosen method not defined')
        
    def _loss_function(self, x, r, y_target):
        x_r = x + r

        kl_loss = tf.keras.backend.categorical_crossentropy(
            y_target, self.f(x_r))

        l2_loss = self.alpha*tf.norm(r, axis=1)

        return tf.reduce_mean(kl_loss + l2_loss) ## Change
    
    def perturb(self):
        start_time = time.time()

        opt = tf.keras.optimizers.legacy.Adam()

        losses = np.zeros(shape=(self.nb_iter, 1))

        n = self.x_test.shape[0]
        self.p_x_test = self.x_test.copy()
        self.p_y_pred = np.zeros(shape=self.y_test.shape, dtype=float)
        self.y_pred = np.zeros(shape=self.y_test.shape, dtype=float)

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

            loss_func = lambda: self._loss_function(x, r, y_target)

            for itr in range(self.nb_iter):
                opt.minimize(loss_func, var_list=[r])
                losses[itr] = losses[itr]+ self._loss_function(x, r, y_target)

            self.p_x_test[cur_i:max_i] = self.p_x_test[cur_i:max_i] + r.numpy()
            self.p_y_pred[cur_i:max_i] = self.f(tf.constant(self.p_x_test[cur_i:max_i],
                                                            dtype=tf.float32)).numpy()
            self.y_pred[cur_i:max_i] = y_pred

        losses = losses / denom
        self.save_loss(losses)

        duration = time.time() - start_time

        df_metrics = self.compute_df_metrics(duration)

        df_metrics.to_csv(os.path.join(self.out_dir, 'df_metrics.csv'))

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

            loss_func = lambda: self._loss_function(x, r, y_target)

            for itr in range(self.nb_iter):
                opt.minimize(loss_func, var_list=[r])
                losses[itr] = losses[itr]+ self._loss_function(x, r, y_target)

            self.p_x_test[cur_i:max_i] = self.p_x_test[cur_i:max_i] + r.numpy()
            self.p_y_pred[cur_i:max_i] = self.f(tf.constant(self.p_x_test[cur_i:max_i],
                                                            dtype=tf.float32)).numpy()
            self.y_pred[cur_i:max_i] = y_pred