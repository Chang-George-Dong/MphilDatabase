import os
import tensorflow as tf
import time
from attacks.attack import Attack
import numpy as np
import matplotlib.pyplot as plt


# Gradient Method (GM)

class Ratio(Attack):
    def __init__(self, model_dir, x_test, y_test, out_dir,
                 eps_init=0.001, nb_iter=1000, smoothness=0.1, mu=1.0, eps=0.1,
                 alpha = 1,lamb = 1, gamma = 1):
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

            target_p_logits = np.ones(shape=y_pred.shape) 

            for i in range(len(y_pred)):
                c = y_pred[i].argmax(axis=0)
                target_p_logits[i, c] = 1.0

            target_p_logits = tf.constant(target_p_logits, dtype=tf.float32)
            return target_p_logits, y_pred
        else:
            raise Exception('Chosen method not defined')
        
    def _loss_function(self, x, r, y_target):
        x_r = x + r

        y_pred = self.f(x_r)
        p_top1, p_top2 = tf.math.top_k(y_pred, 2)[0][:,0], tf.math.top_k(y_pred, 2)[0][:,1]
        p_t = tf.reduce_sum(tf.multiply(y_pred, y_target), axis=-1)

        log_ratio1 = tf.math.log(0.8*p_top1/ p_top2)
        log_ratio2 = tf.math.log(p_top2/ p_t)  # Adding small constant to prevent division by 0
        loss = tf.square(log_ratio1)+tf.square(log_ratio2)   # Adding small constant to prevent log(0)

        return tf.reduce_mean(loss) ## Change
    
    def perturb(self):
        start_time = time.time()

        opt = tf.keras.optimizers.Adam()

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

        opt = tf.keras.optimizers.Adam()

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

        return  self.p_x_test, self.p_y_pred,self.y_pred,
