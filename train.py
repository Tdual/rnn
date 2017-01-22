import numpy as np
import pandas as pd
import tensorflow as tf
import data_helper as dh

max_size = 2000
class_num = 200


#data, one_hots, dic_size = dh.get_tf_data(
#    max_seq_size=max_size, class_num=class_num, max_data_num=2000)
data, one_hots, dic_size = dh.read_tf_data()


data_size = len(one_hots)
train_size = int(data_size*0.9)
test_size = data_size - train_size
print("data size: ", data_size)
print("train size: ", train_size)
print("test size: ", test_size)

rand_idx = np.random.permutation(data_size)
new_data = [data[idx] for idx in rand_idx]
new_one_hots = [one_hots[idx] for idx in rand_idx]
data = new_data
one_hots = new_one_hots
train_x = data[:train_size]
train_y = one_hots[:train_size]
test_x = data[train_size:]
test_y = one_hots[train_size:]



NUM_CLASSES = class_num
NUM_TESTS         = 2000
NUM_EPOCHS        = 10
MINI_BATCH_SIZE    = 64
EMBEDDING_SIZE    = 1280
NUM_FILTERS       = 1280
FILTER_SIZES      = [3,4,5]
L2_LAMBDA         = 0.0001
EVALUATE_EVERY    = 100
CHECKPOINTS_EVERY = 1000
SUMMARY_LOG_DIR = "tmp/tensorflow_log"
CHECKPOINTS_DIR = './'

keep = tf.placeholder(tf.float32)

input_x = tf.placeholder(tf.int32, [None, max_size])
input_y = tf.placeholder(tf.float32, [None, NUM_CLASSES])


with tf.name_scope('embedding'):
    w  = tf.Variable(tf.random_uniform([dic_size, EMBEDDING_SIZE], -1.0, 1.0), name='weight')
    e  = tf.nn.embedding_lookup(w, input_x)
    ex = tf.expand_dims(e, -1)

# Define 3rd and 4th layer (Temporal 1-D convolutional and max-pooling layer).
p_array = []
for filter_size in FILTER_SIZES:
    with tf.name_scope('conv-%d' % filter_size):
        w  = tf.Variable(tf.truncated_normal([ filter_size, EMBEDDING_SIZE, 1, NUM_FILTERS ], stddev=0.02), name='weight')
        b  = tf.Variable(tf.constant(0.1, shape=[ NUM_FILTERS ]), name='bias')
        c0 = tf.nn.conv2d(ex, w, [ 1, 1, 1, 1 ], 'VALID')
        c1 = tf.nn.relu(tf.nn.bias_add(c0, b))
        c2 = tf.nn.max_pool(c1, [ 1,  max_size - filter_size + 1, 1, 1 ], [ 1, 1, 1, 1 ], 'VALID')
        p_array.append(c2)

p = tf.concat(3, p_array)


with tf.name_scope('fc'):
    total_filters = NUM_FILTERS * len(FILTER_SIZES)
    w = tf.Variable(tf.truncated_normal([ total_filters, NUM_CLASSES ], stddev=0.02), name='weight')
    b = tf.Variable(tf.constant(0.1, shape=[ NUM_CLASSES ]), name='bias')
    h0 = tf.nn.dropout(tf.reshape(p, [ -1, total_filters ]), keep)
    predict_y = tf.nn.softmax(tf.matmul(h0, w) + b)


xentropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict_y, input_y))
loss = xentropy + L2_LAMBDA * tf.nn.l2_loss(w)

global_step = tf.Variable(0, name="global_step", trainable=False)
train = tf.train.AdamOptimizer(0.0001).minimize(loss, global_step=global_step)



predict  = tf.equal(tf.argmax(predict_y, 1), tf.argmax(input_y, 1))
accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))

loss_sum   = tf.scalar_summary('train loss', loss)
accr_sum   = tf.scalar_summary('train accuracy', accuracy)
t_loss_sum = tf.scalar_summary('general loss', loss)
t_accr_sum = tf.scalar_summary('general accuracy', accuracy)

saver = tf.train.Saver()


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    writer = tf.train.SummaryWriter(SUMMARY_LOG_DIR, sess.graph)

    train_x_length = len(train_x)
    batch_count = int(train_x_length / MINI_BATCH_SIZE) + 1

    print('Start training.')
    print('     epoch: %d' % NUM_EPOCHS)
    print('mini batch: %d' % MINI_BATCH_SIZE)
    print('train data: %d' % train_x_length)
    print(' test data: %d' % len(test_x))
    print('We will loop %d count per an epoch.' % batch_count)

   
    for epoch in range(NUM_EPOCHS):
        random_indice = np.random.permutation(train_x_length)
        print('Start %dth epoch.' % (epoch + 1))
        for i in range(batch_count):
            mini_batch_x = []
            mini_batch_y = []
            for j in range(min(train_x_length - i * MINI_BATCH_SIZE, MINI_BATCH_SIZE)):
                mini_batch_x.append(train_x[random_indice[i * MINI_BATCH_SIZE + j]])
                mini_batch_y.append(train_y[random_indice[i * MINI_BATCH_SIZE + j]])

            
            _, v1, v2, v3, v4 = sess.run(
                [ train, loss, accuracy, loss_sum, accr_sum ],
                feed_dict={ input_x: mini_batch_x, input_y: mini_batch_y, keep: 0.5 }
            )
            print('%4dth mini batch complete. LOSS: %f, ACCR: %f' % (i + 1, v1, v2))

            current_step = tf.train.global_step(sess, global_step)
            writer.add_summary(v3, current_step)
            writer.add_summary(v4, current_step)


            if current_step % CHECKPOINTS_EVERY == 0:
                saver.save(sess, CHECKPOINTS_DIR + '/model', global_step=current_step)
                print('Checkout was completed.')

 
            if current_step % EVALUATE_EVERY == 0:
                #random_test_indice = np.random.permutation(100)
                #random_test_x = test_x[int(random_test_indice)]
                #random_test_y = test_y[int(random_test_indice)]

                v1, v2, v3, v4 = sess.run(
                    [ loss, accuracy, t_loss_sum, t_accr_sum ],
                    feed_dict={ input_x: test_x, input_y: test_y, keep: 1.0 }
                )
                print('Testing... LOSS: %f, ACCR: %f' % (v1, v2))
                writer.add_summary(v3, current_step)
                writer.add_summary(v4, current_step)


    saver.save(sess, CHECKPOINTS_DIR + '/model-last')
