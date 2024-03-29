from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import inspect
import tensorflow as tf

class Network():
    def __init__(self, config, input_dims, action_dim, master=True):
        self.input_dims = input_dims
        self.action_dim = action_dim
        self.max_moves = config.max_moves
        self.model_name = config.version+'-'\
                            +config.project_name+'_'\
                            +config.method+'_'\
                            +config.model_type+'_'\
                            +config.topology_file+'_'\
                            +config.traffic_file

        self.create_actor_critic_model(config)

        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                config.initial_learning_rate,
                config.learning_rate_decay_step,
                config.learning_rate_decay_rate,
                staircase=True)

        if config.optimizer == 'RMSprop':
            self.actor_optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.lr_schedule)
            self.critic_optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.lr_schedule)
        elif config.optimizer == 'Adam':
            self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
            self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)

        if master:
            self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), actor_optimizer=self.actor_optimizer, critic_optimizer=self.critic_optimizer, model=self.model)
            self.ckpt_dir = './tf_ckpts/'+self.model_name
            self.manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_dir, max_to_keep=config.max_to_keep)
            self.writer = tf.compat.v2.summary.create_file_writer('./logs/%s' % self.model_name)
            #self.save_hyperparams(config)
            self.model.summary()

    def create_actor_critic_model(self, config):
        tf.keras.backend.set_floatx('float32')
        inputs = tf.keras.Input(shape=(self.input_dims[0], self.input_dims[1], self.input_dims[2]))
       
        # Actor
        if config.l2_regularizer > 0:
            Conv2D_1 = tf.keras.layers.Conv2D(config.Conv2D_out, 3, padding='same', use_bias=not config.batch_norm, 
                    kernel_regularizer=tf.keras.regularizers.l2(config.l2_regularizer))
        else:
            Conv2D_1 = tf.keras.layers.Conv2D(config.Conv2D_out, 3, padding='same', use_bias=not config.batch_norm) 
        x_1 = Conv2D_1(inputs)
        if config.batch_norm:
            x_1 = tf.keras.layers.BatchNormalization()(x_1)
        x_1 = tf.keras.layers.LeakyReLU()(x_1)
        x_1 = tf.keras.layers.Flatten()(x_1)
        if config.l2_regularizer > 0:
            Dense1_1 = tf.keras.layers.Dense(config.Dense_out, kernel_regularizer=tf.keras.regularizers.l2(config.l2_regularizer))
        else:
            Dense1_1 = tf.keras.layers.Dense(config.Dense_out)
        x_1 = Dense1_1(x_1)
        x_1 = tf.keras.layers.LeakyReLU()(x_1)
        #Dense2_1 = tf.keras.layers.Dense(self.action_dim, activation='softmax')
        Dense2_1 = tf.keras.layers.Dense(self.action_dim)
        logits = Dense2_1(x_1)
        #Logit clipping
        if config.logit_clipping > 0:
            logits = config.logit_clipping*tf.keras.activations.tanh(logits)

        # Critic
        if config.l2_regularizer > 0:
            Conv2D_2 = tf.keras.layers.Conv2D(config.Conv2D_out, 3, padding='same', use_bias=not config.batch_norm, 
                    kernel_regularizer=tf.keras.regularizers.l2(config.l2_regularizer))
        else:
            Conv2D_2 = tf.keras.layers.Conv2D(config.Conv2D_out, 3, padding='same', use_bias=not config.batch_norm) 
        x_2 = Conv2D_2(inputs)
        if config.batch_norm:
            x_2 = tf.keras.layers.BatchNormalization()(x_2)
        x_2 = tf.keras.layers.LeakyReLU()(x_2)
        x_2 = tf.keras.layers.Flatten()(x_2)
        if config.l2_regularizer > 0:
            Dense1_2 = tf.keras.layers.Dense(config.Dense_out, kernel_regularizer=tf.keras.regularizers.l2(config.l2_regularizer))
        else:
            Dense1_2 = tf.keras.layers.Dense(config.Dense_out)
        x_2 = Dense1_2(x_2)
        x_2 = tf.keras.layers.LeakyReLU()(x_2)
        if config.normalized_reward:
            Dense2_2 = tf.keras.layers.Dense(1, activation='sigmoid')
        else:
            Dense2_2 = tf.keras.layers.Dense(1)
        values = Dense2_2(x_2)

        self.model = tf.keras.models.Model(inputs, [logits, values])

        self.actor_model = tf.keras.models.Model(inputs, logits)
        self.critic_model = tf.keras.models.Model(inputs, values)

    def value_loss_fn(self, rewards, values):
        advantages = tf.convert_to_tensor(rewards[:, None], dtype=tf.float32) - values
        value_loss = advantages ** 2
        value_loss = tf.reduce_mean(value_loss)

        #regularization_loss = tf.math.add_n(self.critic_model.losses)
        #value_loss += regularization_loss
        
        return value_loss, advantages

    def policy_loss_fn_action_combination(self, logits, actions, advantages, entropy_weight=0.01, log_epsilon=1e-12):
        actions = tf.reshape(actions, [-1, self.max_moves, self.action_dim])                            #actions shape = [batch_size, max_moves, action_dim]
        policy = tf.nn.softmax(logits)                                                                  #policy shape = [batch_size, action_dim]
        assert policy.shape[0] == actions.shape[0] and advantages.shape[0] == actions.shape[0]                  
        entropy = tf.nn.softmax_cross_entropy_with_logits(labels=policy, logits=logits)                 #entropy already includes "-", shape = [batch_size,]
        entropy = tf.expand_dims(entropy, -1)                                                           #[batch_size, 1]
        policy = tf.expand_dims(policy, -1)                                                             #policy shape = [batch_size, action_dim, 1]
        policy_loss = tf.math.log(tf.maximum(tf.squeeze(tf.matmul(actions, policy)), log_epsilon))      #[batch_size, max_moves]
        policy_loss = tf.reduce_sum(policy_loss, 1, keepdims=True)                                      #[batch_size, 1]
        """ '-'advantages means gradient asdcend. same for entropy """
        policy_loss = tf.multiply(policy_loss, tf.stop_gradient(-advantages))                           #[batch_size, 1]
        policy_loss -= entropy_weight * entropy
        policy_loss = tf.reduce_sum(policy_loss)

        #regularization_loss = tf.math.add_n(self.actor_model.losses)
        #policy_loss += regularization_loss
        
        return policy_loss, entropy

    @tf.function
    def actor_critic_train(self, inputs, actions, rewards, entropy_weight=0.01):
        # Tracks the variables involved in computing the loss by using tf.GradientTape
        with tf.GradientTape() as tape:
            values = self.critic_model(inputs, training=True)
            value_loss, advantages = self.value_loss_fn(rewards, values)

        critic_gradients = tape.gradient(value_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic_model.trainable_variables))

        with tf.GradientTape() as tape:
            logits = self.actor_model(inputs, training=True)
            policy_loss, entropy = self.policy_loss_fn_action_combination(logits, actions, advantages, entropy_weight)

        actor_gradients = tape.gradient(policy_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor_model.trainable_variables))

        return value_loss, entropy, actor_gradients, critic_gradients

    @tf.function
    def actor_predict(self, inputs, softmax_temp=1):
        logits = self.actor_model(inputs, training=False)
        policy = tf.nn.softmax(logits/softmax_temp)

        return policy
 
    def restore_ckpt(self, checkpoint=''):
        if checkpoint == '':
            checkpoint = self.manager.latest_checkpoint
        else:
            checkpoint = self.ckpt_dir+'/'+checkpoint

        self.ckpt.restore(checkpoint).expect_partial()
        if checkpoint:
            step = int(self.ckpt.step)
            print("Restored from {}".format(checkpoint), step)
        else:
            step = 0
            print("Initializing from scratch.")

        return step

    def save_ckpt(self, _print=False):
        save_path = self.manager.save()
        if _print:
            print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))

    def inject_summaries(self, summary_dict, step):
        with self.writer.as_default():
            for summary in summary_dict:
                tf.summary.scalar(summary, summary_dict[summary], step=step)
            self.writer.flush()

    def save_hyperparams(self, config):
        fp = self.ckpt_dir + '/hyper_parameters'

        hparams = {k:v for k, v in inspect.getmembers(config)
            if not k.startswith('__') and not callable(k)}

        if os.path.exists(fp):
            f = open(fp, 'r')
            match = True
            for line in f:
                idx = line.find('=')
                if idx == -1:
                    continue
                k = line[:idx-1]
                v = line[idx+2:-1]
                if v != str(hparams[k]):
                    match = False
                    print('[!] Unmatched hyperparameter:', k, v, hparams[k])
                    break
            f.close()
            if match:
                return
            
            f = open(fp, 'a')
        else:
            if not os.path.exists(self.ckpt_dir):
                os.makedirs(self.ckpt_dir)
            f = open(fp, 'w+')

        for k, v in hparams.items():
            f.writelines(k + ' = ' + str(v) + '\n')
        f.writelines('\n')
        print("Save hyper parameters: %s" % fp)
        f.close()
