import tensorflow as tf
import numpy as np
import scipy.misc
import random



def readMask(filename):
    return scipy.misc.imread(filename)

def add_curr_State(curr_State):
    experience_Replay.add(curr_State)

def get_Next_Coordinates(Pos_temp, curr_action):
    if curr_action == 'left':
        Pos_temp[1] -= 1
    elif curr_action == 'right':
        Pos_temp[1] += 1
    elif curr_action == 'up':
        Pos_temp[0] -= 1
    elif curr_action == 'down':
        Pos_temp[0] += 1
    return Pos_temp




def DNN_fullyConnected2(frame):

    fc1 = tf.reshape(frame, shape=[-1, mn])

    w1 = tf.get_variable('weights1',
                         shape=[mn, 600],
                         dtype=tf.float32,
                         initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
    b1 = tf.get_variable('biases1',
                         shape=[600],
                         dtype=tf.float32,
                         initializer=tf.constant_initializer(0))
    fc2 = tf.nn.relu(tf.matmul(fc1, w1) + b1)


    w3 = tf.get_variable('weights3',
                         shape=[600, 600],
                         dtype=tf.float32,
                         initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
    b3 = tf.get_variable('biases3',
                         shape=[600],
                         dtype=tf.float32,
                         initializer=tf.constant_initializer(0))
    fc4 = tf.nn.relu(tf.matmul(fc2, w3) + b3)


    w4 = tf.get_variable('weights4',
                         shape=[600, 4],
                         dtype=tf.float32,
                         initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
    b4 = tf.get_variable('biases4',
                         shape=[4],
                         dtype=tf.float32,
                         initializer=tf.constant_initializer(0))
    output = tf.matmul(fc4, w4) + b4




    return output


def DNN_fullyConnected(frame):

    fc1 = tf.reshape(frame, shape=[-1, mn])

    w1 = tf.get_variable('weights1',
                         shape=[mn, 8000],
                         dtype=tf.float32,
                         initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
    b1 = tf.get_variable('biases1',
                         shape=[8000],
                         dtype=tf.float32,
                         initializer=tf.constant_initializer(0))
    fc2 = tf.nn.relu(tf.matmul(fc1, w1) + b1)

    w2 = tf.get_variable('weights2',
                         shape=[8000, 4000],
                         dtype=tf.float32,
                         initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
    b2 = tf.get_variable('biases2',
                         shape=[4000],
                         dtype=tf.float32,
                         initializer=tf.constant_initializer(0))
    fc3 = tf.nn.relu(tf.matmul(fc2, w2) + b2)

    w3 = tf.get_variable('weights3',
                         shape=[4000, 4000],
                         dtype=tf.float32,
                         initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
    b3 = tf.get_variable('biases3',
                         shape=[4000],
                         dtype=tf.float32,
                         initializer=tf.constant_initializer(0))
    fc4 = tf.nn.relu(tf.matmul(fc3, w3) + b3)


    w4 = tf.get_variable('weights4',
                         shape=[4000, 4],
                         dtype=tf.float32,
                         initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
    b4 = tf.get_variable('biases4',
                         shape=[4],
                         dtype=tf.float32,
                         initializer=tf.constant_initializer(0))
    output = tf.matmul(fc4, w4) + b4




    return output





def CNN_DeepLearning(frame):
    w1 = tf.get_variable('weight1',
                        shape=[8, 8, 1, 64],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    b1 = tf.get_variable('bias1',
                        shape=[64],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv1 = tf.nn.conv2d(frame, w1,
                         strides=[1, 4, 4, 1],
                         padding='VALID')
    conv1 = tf.nn.bias_add(conv1, b1)

    conv1 = tf.nn.relu(conv1)


    w2 = tf.get_variable('weight2',
                        shape=[4, 4, 64, 128],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    b2 = tf.get_variable('bias2',
                        shape=[128],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv2 = tf.nn.conv2d(conv1, w2,
                         strides=[1, 2, 2, 1],
                         padding='VALID')
    conv2 = tf.nn.bias_add(conv2, b2)

    conv2 = tf.nn.relu(conv2)


    w3 = tf.get_variable('weight3',
                        shape=[3, 3, 128, 128],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    b3 = tf.get_variable('bias3',
                        shape=[128],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0))
    conv3 = tf.nn.conv2d(conv2, w3,
                         strides=[1, 1, 1, 1],
                         padding='VALID')
    conv3 = tf.nn.bias_add(conv3, b3)

    conv3 = tf.nn.relu(conv3)


    fc4 = tf.reshape(conv3, shape=[-1, 6272])

    # dim = fc4.get_shape()[1].value

    w4 = tf.get_variable('weights4',
                         shape=[6272, 1024],
                         dtype=tf.float32,
                         initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    b4 = tf.get_variable('biases4',
                         shape=[1024],
                         dtype=tf.float32,
                         initializer=tf.constant_initializer(0))
    fc5 = tf.nn.relu(tf.matmul(fc4, w4) + b4)

    w5 = tf.get_variable('weights5',
                         shape=[1024, 4],
                         dtype=tf.float32,
                         initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    b5 = tf.get_variable('biases5',
                         shape=[4],
                         dtype=tf.float32,
                         initializer=tf.constant_initializer(0))
    output = tf.matmul(fc5, w5) + b5

    return output


def loss_func(y, output, action):

    output1 = tf.multiply(output, action)
    output2 = tf.reduce_sum(output1, 1)
    loss = tf.reduce_sum(tf.square(y - output2))
    return loss

def training(loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    return optimizer



# initialization
curr_Pos = np.array([0,0])
curr_Pos_tuple = tuple(curr_Pos)    #current position
filename = 'D:\\myPythonProjects\\samplingTrajectory\\mask1_41.png'
mask_Map = readMask(filename)
n, m = mask_Map.shape
mn = mask_Map.size
curr_Map = np.array(mask_Map)/255
ob = np.copy(curr_Map)
ob[4][1] = 2
ob = np.reshape(ob, [1, n, m ,1])
curr_Map[curr_Pos[0], curr_Pos[1]] = 0
next_Map = np.copy(curr_Map)
curr_Map_tuple = tuple([tuple(row.astype('uint8')) for row in curr_Map]) #current map
ACTIONS = ['left', 'right', 'up', 'down']
future_Discount = 0.92   #future reward discount
epsilion = 0.75      #will to explore unknowns
experience_Replay = set()
total_Episode = 10000000000000000
Mini_Match_Size = 600
learning_rate = 0.00005
save_cnt = 1

OUTBOUND_PENALTY = -5



x = tf.placeholder(tf.float32, [None, n, m, 1])
y = tf.placeholder(tf.float32, [None])
action = tf.placeholder(tf.float32, [None, 4])


# output = CNN_DeepLearning(x)
output = DNN_fullyConnected2(x)
train_loss = loss_func(y, output, action)
train_optimizer = training(train_loss, learning_rate)
saver = tf.train.Saver()

with tf.Session() as sess:


    sess.run(tf.global_variables_initializer())


    for Episode in range(total_Episode):
        # initaliza search
        curr_Pos = np.array([int(n / 2), int(m / 2)])
        curr_Map = (np.array(mask_Map)/255).astype('uint8')
        curr_Map[curr_Pos[0], curr_Pos[1]] = 0
        next_Map = np.copy(curr_Map)

        # curr_Pos_tuple = tuple(curr_Pos)
        # curr_Map_tuple = tuple([tuple(row.astype('uint8')) for row in curr_Map])
        # add_curr_State(curr_Map_tuple + curr_Pos_tuple)

        is_Terminal = False




        while not is_Terminal:
            # incentive to explore

            next_Map = np.copy(curr_Map)


            if (np.random.uniform() > epsilion):
                curr_action = np.random.choice(ACTIONS)
            else:
                curr_Map_feed = np.copy(curr_Map)
                curr_Map_feed[curr_Pos[0]][curr_Pos[1]] = 2
                curr_Map_feed = np.reshape(curr_Map_feed, [1, n, m, 1])
                curr_action = sess.run(output, feed_dict={x:curr_Map_feed})
                curr_action = np.argmax(curr_action)
                curr_action = ACTIONS[int(curr_action)]


            next_Pos = np.copy(curr_Pos)
            get_Next_Coordinates(next_Pos, curr_action)

            if next_Pos[0] < 0 or next_Pos[1] < 0 or next_Pos[0] >= n or next_Pos[1] >= m:
                is_Terminal = True
                reward = OUTBOUND_PENALTY
            else:
                if curr_Map[next_Pos[0], next_Pos[1]] == 1:
                    reward = 200
                else:
                    reward = 0
                next_Map[next_Pos[0], next_Pos[1]] = 0

            if np.sum(next_Map) == 0:
                is_Terminal = True


            curr_Pos_tuple = tuple(curr_Pos)
            curr_Map_tuple = tuple([tuple(row) for row in curr_Map])
            action_tuple = tuple(np.array([ACTIONS.index(curr_action)]))
            reward_tuple = tuple([reward])
            add_curr_State(curr_Map_tuple + curr_Pos_tuple + action_tuple + reward_tuple)

            train_data = random.sample(experience_Replay, np.minimum(Mini_Match_Size, len(experience_Replay)))
            Q_value = np.empty([0])
            # Q_value_curr_step = np.empty(0)
            # Q_value = np.append([Q_value], [5])
            # train_map = np.empty([1, n, m, 1])
            train_map = np.empty([0, n, m, 1])
            action_data = np.empty([0, 4])


            for train_data_element in train_data:
                is_Terminal_data = False
                curr_Pos[0] = train_data_element[n]
                curr_Pos[1] = train_data_element[n+1]
                curr_action = ACTIONS[int(train_data_element[n+2])]
                reward = train_data_element[n+3]
                temp_Map = train_data_element[0:n][0:m]
                temp_Map = np.array(temp_Map)
                curr_Pos_temp = np.copy(curr_Pos)
                get_Next_Coordinates(curr_Pos_temp, curr_action)
                if curr_Pos_temp[0] < 0 or curr_Pos_temp[1] < 0 or curr_Pos_temp[0] >= n or curr_Pos_temp[1] >= m:
                    is_Terminal_data = True
                else:
                    temp_Map[curr_Pos_temp[0],curr_Pos_temp[1]] = 0
                    if np.sum(temp_Map) == 0:
                        is_Terminal_data = True

                if is_Terminal_data:
                    Q_value = np.append(Q_value, reward)
                else:
                    temp_Map[[curr_Pos_temp[0]], [curr_Pos_temp[1]]] = 2
                    temp_Map = np.reshape(temp_Map, [1, n, m, 1])
                    future_reward = sess.run(output, feed_dict={x:temp_Map})

                    reward = reward + np.amax(future_reward) * future_Discount
                    Q_value = np.append(Q_value, reward)

                temp_Map = train_data_element[0:n][0:m]
                temp_Map = np.array(temp_Map)
                temp_Map[curr_Pos[0], curr_Pos[1]] = 2
                #Q_temp = sess.run(output, feed_dict={x:temp_Map})
                #Q_temp = Q_temp[np.array([ACTIONS.index(curr_action)])]
                #Q_value_curr_step = np.append([Q_value_curr_step], [Q_temp])
                temp_Map = np.reshape(temp_Map, [1, n, m, 1])
                train_map = np.append(train_map, temp_Map, axis=0)
                temp = np.zeros([1, 4])
                temp[0, ACTIONS.index(curr_action)] = 1
                action_data = np.row_stack((action_data, temp))


            _, tra_loss = sess.run([train_optimizer, train_loss], feed_dict={x:train_map, action:action_data, y:Q_value})
            #print(tra_loss)


            if not is_Terminal:
                curr_Map[next_Pos[0], next_Pos[1]] = 0
                curr_Pos = np.copy(next_Pos)



        temp_ob = sess.run(output, feed_dict={x: ob})
        print(temp_ob)

        # print(temp_ob)
        if (Episode)% 200 == 0:
            print(Episode)
            print(len(experience_Replay))
            save_path = saver.save(sess, "D:\\myPythonProjects\\ModelParameters\\scanTrajectory_" + str(n) + "_3_" + str(save_cnt) + ".ckpt")
            save_cnt += 1
            print("Save to path:", save_path)













