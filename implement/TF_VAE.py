import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
fully_connected =  tf.contrib.layers.fully_connected 

# ref: https://arxiv.org/abs/1312.6114

def main():
    n_ipts = 28 * 28
    n_opts = n_ipts
    n_unit = [500, 500, 20, 500, 500]
    n_digits = 60
    n_epochs = 50
    batch_size = 150
    learning_rate = 0.001

    mnist = input_data.read_data_sets("../_data/")

    with tf.contrib.framework.arg_scope(
        [fully_connected],
        activation_fn = tf.nn.elu,
        weights_initializer = tf.contrib.layers.variance_scaling_initializer()
    ):
        X = tf.placeholder(tf.float32, [None, n_ipts])
        hidden1 = fully_connected(X, n_unit[0])
        hidden2 = fully_connected(hidden1, n_unit[1])
        hidden3_mean = fully_connected(hidden2, n_unit[2], activation_fn=None)
        hidden_gamma = fully_connected(hidden2, n_unit[2], activation_fn=None)
        hidden3_sigma = tf.exp(0.5 * hidden_gamma)
        noise = tf.random_normal(tf.shape(hidden3_sigma), dtype=tf.float32)
        hidden3 = hidden3_mean + hidden3_sigma * noise
        hidden4 = fully_connected(hidden3, n_unit[3])
        hidden5 = fully_connected(hidden4, n_unit[4])
        logits = fully_connected(hidden5, n_opts, activation_fn=None)
        output = tf.sigmoid(logits)
    
    with tf.name_scope('train_op'):
        reconstruction_loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=logits)
        )
        latent_loss = tf.reduce_sum(
            tf.exp(hidden_gamma) + tf.square(hidden3_mean) - hidden_gamma - 1
        )
        loss = reconstruction_loss + latent_loss
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epochs):
            n_batches = mnist.train.num_examples // batch_size
            for _ in range(n_batches):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                l, _ = sess.run([loss, train_op], feed_dict={X: X_batch})
                print('Loss {} in epoch - {}'.format(l, epoch))
        codings_rnd = np.random.normal(size=[n_digits, n_unit[2]])
        outputs_val = output.eval(feed_dict={hidden3: codings_rnd})

    for iteration in range(n_digits):
        plt.subplot(n_digits, 10, iteration + 1)
        plt.imshow(outputs_val[iteration].reshape([28, 28]), cmap="Greys", interpolation="nearest")
        plt.axis("off")
        # plot_image(outputs_val[iteration])
    plt.show()

if __name__ == '__main__':
    main()