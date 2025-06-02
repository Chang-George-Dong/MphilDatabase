import os
import tensorflow as tf
import time
from attacks.attack import Attack
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.stats import entropy

# Gradient Method (GM)

class GM_kl(Attack):
    def __init__(self, model_dir, x_test, y_test, out_dir,
                 eps_init=0.001, nb_iter=1000, smoothness=0.1, mu=1.0, eps=0.1, theta = 0.01):
        super().__init__()
        self.f = self._model_as_tf_function(tf.keras.models.load_model(model_dir))
        self.x_test = x_test
        self.y_test = y_test  # used for evaluation
        self.eps_init = eps_init
        self.nb_iter = nb_iter
        self.out_dir = out_dir
        self.eps = eps  # max allowed perturbation
        self.mu = mu
        self.p_x_test = None
        self.p_y_pred = None
        self.y_pred = None
        self.theta = theta

    def _get_y_target(self, x, method='average_case'):
        """
        :param x: the input tensor
        :param method: how you want to generate the target
        :return: the target predictions, the predictions for the unperturbed data
        """
        if method == 'average_case':
            y_pred = self.f(x)
            y_pred = tf.stop_gradient(y_pred).numpy()

            target_p_logits = np.ones(shape=y_pred.shape) * 1e-5

            for i in range(len(y_pred)):
                c = y_pred[i].argmax(axis=0)
                c_s = list(range(y_pred.shape[1]))
                c_s.remove(c)
                new_c = np.random.choice(c_s)
                target_p_logits[i, new_c] = 1.0

            target_p_logits = tf.constant(target_p_logits, dtype=tf.float32)
            return tf.nn.softmax(target_p_logits), y_pred
        else:
            raise Exception('Chosen method not defined')
        
    def _loss_function(self, x, r, y_target):
        x_r = x + r
        kl_loss = self.theta * tf.keras.losses.KLDivergence()(y_target, self.f(x_r))
        ce_loss = tf.keras.backend.categorical_crossentropy(
            y_target, self.f(x_r))


        return tf.reduce_mean(ce_loss + kl_loss) ## Change

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
                losses[itr] = losses[itr] + self._loss_function(x, r, y_target)

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
                        
            #用于储存
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

                loss_func = lambda: self._loss_function(x, r, y_target)


                for itr in range(self.nb_iter):
                    opt.minimize(loss_func, var_list=[r])
                    losses[itr] = losses[itr] + self._loss_function(x, r, y_target)

                    px_test[cur_i:max_i] = self.x_test[cur_i:max_i].copy() + r.numpy()
                    py_test[cur_i:max_i] = self.f(tf.constant(px_test[cur_i:max_i],
                                                                dtype=tf.float32)).numpy()
                    

                self.p_x_test[cur_i:max_i] = self.p_x_test[cur_i:max_i] + r.numpy()
                self.p_y_pred[cur_i:max_i] = self.f(tf.constant(self.p_x_test[cur_i:max_i],
                                                                dtype=tf.float32)).numpy()
                self.y_pred[cur_i:max_i] = y_pred

            return self.p_x_test, self.p_y_pred, self.y_pred
    
    def draw(self):
            start_time = time.time()

            opt = tf.keras.optimizers.legacy.Adam()

            losses = np.zeros(shape=(self.nb_iter, 1))

            n = self.x_test.shape[0]
            self.p_x_test = self.x_test.copy()
            self.p_y_pred = np.zeros(shape=self.y_test.shape, dtype=float)
            self.y_pred = np.zeros(shape=self.y_test.shape, dtype=float)

            denom = 0
                        
            #用于储存
            

            kl_result = []
            dist_result = []
            attack_result = []
            js_result = []

            def calculate_kl(p,q):

                cross_entropy = entropy(p, q, axis =1)
                entropy_p = entropy(p, axis =1)

                return cross_entropy - entropy_p


            def calculate_js(y_before, y_after):
                # 计算M
                M = 0.5 * (y_before + y_after)
                # 计算JSD
                return 0.5 * entropy(y_before, M, axis =1) + 0.5 * entropy(y_after, M, axis =1)

            def calculate_success(y_before, y_after):
                return np.argmax(y_before,axis = 1) != np.argmax(y_after,axis = 1)
            
            def calculate_dist(x_before, x_after):
                return np.sqrt(np.sum((x_before-x_after)**2,axis =1))
            
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
                px_test = self.x_test.copy()
                py_test = y_pred.copy()

                loss_func = lambda: self._loss_function(x, r, y_target)


                for itr in range(self.nb_iter):

                    kl_divergence = calculate_kl(y_pred,py_test)
                    js_divergence = calculate_js(y_pred,py_test)
                    distance = calculate_dist(px_test,self.x_test)
                    attack_success = calculate_success(py_test,y_pred)

                    kl_result.append(kl_divergence)
                    dist_result.append(distance)
                    attack_result.append(attack_success)
                    js_result.append(js_divergence)

                    opt.minimize(loss_func, var_list=[r])
                    losses[itr] = losses[itr] + self._loss_function(x, r, y_target)

                    px_test[cur_i:max_i] = self.x_test[cur_i:max_i].copy() + r.numpy()
                    py_test[cur_i:max_i] = self.f(tf.constant(px_test[cur_i:max_i],
                                                                dtype=tf.float32)).numpy()
                    

                self.p_x_test[cur_i:max_i] = self.p_x_test[cur_i:max_i] + r.numpy()
                self.p_y_pred[cur_i:max_i] = self.f(tf.constant(self.p_x_test[cur_i:max_i],
                                                                dtype=tf.float32)).numpy()
                self.y_pred[cur_i:max_i] = y_pred

            return self.p_x_test, self.p_y_pred, self.y_pred,kl_result, js_result, dist_result, attack_result