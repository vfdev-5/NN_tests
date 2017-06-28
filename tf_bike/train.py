import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


class Trainer:

    def __init__(self, log_dir="logs"):
        self._log_dir = log_dir
        self.display_step = 100

    @staticmethod
    def get_batch(batch_index, batch_size, train_x, train_y):
        start = batch_size * batch_index
        end = batch_size * (batch_index + 1)
        return train_x[start:end, ...], train_y[start:end, ...]

    def train(self, trainval_x, trainval_y, training_params, verbose=0):
        """
        NN train method
        :param trainval_x:
        :param trainval_y:
        :param training_params: a dictionary with training parameters:

        :param verbose: verbose level: 0, 1, 2 etc
        :return:
        """
        assert isinstance(trainval_x, np.ndarray), "trainval_x should be an ndarray"
        assert isinstance(trainval_y, np.ndarray), "trainval_y should be an ndarray"
        assert isinstance(training_params, dict), "training_params should be a dict"

        assert 'network' in training_params and callable(training_params['network']), \
            "training_params should contain network function"
        assert 'loss' in training_params and callable(training_params['loss']), \
            "training_params should contain loss function"
        assert 'optimizer' in training_params, \
            "training_params should contain optimizer function"
        assert 'training_epochs' in training_params, \
            "training_params should contain training_epochs key"

        network_f = training_params['network']
        loss_f = training_params['loss']
        optimizer_f = training_params['optimizer']
        lr = training_params['lr']
        lr_kwargs = None if 'lr_kwargs' not in training_params else training_params['lr_kwargs']
        metrics_list = None if 'metrics' not in training_params else training_params['metrics']
        batch_size = training_params['batch_size'] if 'batch_size' in training_params else 16

        n_features = np.prod(trainval_x.shape[1:])
        n_classes = np.prod(trainval_y.shape[1:])

        tf.reset_default_graph()
        X = tf.placeholder(tf.float32, shape=(batch_size, n_features), name="X")
        Y_true = tf.placeholder(tf.float32, shape=(batch_size, n_classes), name="Y_true")

        with tf.name_scope('Model'):
            Y_pred = network_f(X)

        with tf.name_scope('Loss'):
            loss = loss_f(Y_true, Y_pred)
        tf.summary.scalar("loss", loss)

        with tf.name_scope('Optimizer'):
            global_step = tf.Variable(0, trainable=False, name='global_step')
            if callable(lr):
                if lr_kwargs is not None:
                    lr_f = lr(global_step, **lr_kwargs)
                else:
                    lr_f = lr(global_step)
            else:
                lr_f = lr
            optimizer = optimizer_f(lr_f).minimize(loss, global_step=global_step)

        metrics = []
        if metrics_list is not None:
            if not isinstance(metrics, list):
                metrics = [metrics_list,]
            for i, m in enumerate(metrics_list):
                assert isinstance(m, (tuple, list)) and len(m) == 2, \
                    "Metric should be a tuple ('metric_name', function)"
                metric_name = m[0]
                with tf.name_scope(metric_name):
                    metrics.append(m[1](Y_true, Y_pred))
                tf.summary.scalar(metric_name, metrics[-1])

        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()

        if verbose > 0:
            print("Start training")

        train_path = os.path.join(self._log_dir, 'train')
        val_path = os.path.join(self._log_dir, 'val')

        if not os.path.exists(train_path):
            os.makedirs(train_path)
        if not os.path.exists(val_path):
            os.makedirs(val_path)

        seed = np.random.randint(0, 10000) if 'seed' not in training_params else training_params['seed']
        tf.set_random_seed(seed)

        model_name = Y_pred.name[:-2]
        saver_model_path_templ = os.path.join(self._log_dir,
                                              '%s_seed=%i_loss={loss}_val_loss={val_loss}' % (model_name, seed))

        sv = tf.train.Supervisor(logdir=train_path,
                                 summary_op=merged_summary_op,
                                 checkpoint_basename=model_name + '.ckpt')
        # saver = sv.saver

        with sv.managed_session() as sess:

            train_writer = sv.summary_writer
            val_writer = tf.summary.FileWriter(val_path)

            train_size = training_params['train_size'] if 'train_size' in training_params else 0.75

            if trainval_x.dtype != np.float32:
                trainval_x = trainval_x.astype(np.float32)
            if trainval_y.dtype != np.float32:
                trainval_y = trainval_y.astype(np.float32)
            train_x, val_x, train_y, val_y = train_test_split(trainval_x, trainval_y, train_size=train_size)

            training_epochs = training_params['training_epochs']
            train_ops = [optimizer, loss, merged_summary_op]
            train_ops.extend(metrics)
            val_ops = [loss, merged_summary_op]
            val_ops.extend(metrics)

            def run_epoch(ops, _x, _y, _writer, is_training_phase):
                avg_loss = 0.0
                avg_metrics = np.zeros((len(metrics),))
                n_batchs = int(_x.shape[0] / batch_size)

                prefix = 'val_' if not is_training_phase else ''
                ops_loss_index = 0 if not is_training_phase else 1

                # Train over all batches
                for i in range(n_batchs):
                    if verbose > 1:
                        print("-- %i / %i" % (i, n_batchs))

                    batch_x, batch_y = Trainer.get_batch(i, batch_size, _x, _y)
                    # Run optimization op (backprop), loss op (to get loss value)
                    # and summary nodes
                    ret = sess.run(ops, feed_dict={X: batch_x, Y_true: batch_y})
                    loss_value = ret[ops_loss_index]
                    summary = ret[ops_loss_index+1]
                    metrics_values = ret[ops_loss_index+2:]

                    # Write logs at every iteration
                    _writer.add_summary(summary, epoch * n_batchs + i)

                    # Compute average loss
                    avg_loss += loss_value * 1.0 / n_batchs
                    for j, v in enumerate(metrics_values):
                        avg_metrics[j] += v * 1.0 / n_batchs

                        # Display logs per epoch step
                if verbose > 0 and (epoch + 1) % self.display_step == 0:
                    print(prefix + "loss=%.9f | " % avg_loss, end='')
                    for (name, _), avg_value in zip(metrics_list, avg_metrics):
                        print(prefix + "%s=%.9f" % (name, avg_value), end=' ')
                    print('')

                return avg_loss, avg_metrics

            # def save_model(_train_loss, _val_loss):
            #     saver.save(sess, save_path=saver_model_path_templ.format(loss=_train_loss, val_loss=_val_loss))

            # Training cycle
            best_avg_val_loss = None
            train_avg_loss = None
            train_avg_metrics = None
            val_avg_loss = None
            val_avg_metrics = None
            for epoch in range(training_epochs):
                if sv.should_stop():
                    break

                if verbose > 0 and (epoch + 1) % self.display_step == 0:
                    print("Epoch: %04d" % (epoch + 1))

                train_avg_loss, train_avg_metrics = run_epoch(train_ops,
                                                              train_x, train_y,
                                                              train_writer, is_training_phase=True)
                val_avg_loss, val_avg_metrics = run_epoch(val_ops,
                                                          val_x, val_y,
                                                          val_writer, is_training_phase=False)
                # if best_avg_val_loss is None or best_avg_val_loss > val_avg_loss:
                #     best_avg_val_loss = val_avg_loss
                #     save_model(train_avg_loss, val_avg_loss)

            if verbose > 0:
                print("Optimization Finished!")

            # Save the last model
            # save_model(train_avg_loss, val_avg_loss)


