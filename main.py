import numpy as np
import tensorflow as tf
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TESTDATADIR = "/home/gk99/Apps/python_env/venv/birdsDataset/archive/test"
TRAINDATADIR = "/home/gk99/Apps/python_env/venv/birdsDataset/archive/train"
VALDATADIR = "/home/gk99/Apps/python_env/venv/birdsDataset/archive/valid"

data_list = []
label_list = []

base_model = tf.keras.applications.ResNet101V2(input_shape = (80,80,3),
                                               include_top = False,
                                               weights = "imagenet")
base_model.trainable = False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
disc_layer1 = tf.keras.layers.Dense(1024, activation = "relu")
disc_layer2 = tf.keras.layers.Dense(512, activation = "relu")
prediction_layer = tf.keras.layers.Dense(43, activation = "softmax")


model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    disc_layer1,
    disc_layer2,
    prediction_layer
])




model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


for cat in os.listdir(TRAINDATADIR):
    cat_path = os.path.join(TRAINDATADIR, str(cat))
    for img in os.listdir(cat_path):
        entry = cv2.imread(cat_path + '/' + str(img))
        entry = entry / 255.0
        entry = cv2.resize(entry, (80,80), interpolation=cv2.INTER_NEAREST)
        data_list.append(entry)
        label_list.append(cat)
x_train= np.array(data_list)
y_train = np.array(label_list)

data_list.clear()
label_list.clear()

for cat in os.listdir(VALDATADIR):
    cat_path = os.path.join(VALDATADIR, str(cat))
    for img in os.listdir(cat_path):
        entry = cv2.imread(cat_path + '/' + str(img))
        entry = entry / 255.0
        entry = cv2.resize(entry, (80,80), interpolation=cv2.INTER_NEAREST)
        data_list.append(entry)
        label_list.append(cat)
x_validate = np.array(data_list)
y_validate = np.array(label_list)

data_list.clear()
label_list.clear()


for cat in os.listdir(TESTDATADIR):
    cat_path = os.path.join(TESTDATADIR, str(cat))
    for img in os.listdir(cat_path):
        entry = cv2.imread(cat_path + '/' + str(img))
        entry = entry / 255.0
        entry = cv2.resize(entry, (80,80), interpolation=cv2.INTER_NEAREST)
        data_list.append(entry)
        label_list.append(cat)
x_test = np.array(data_list)
y_test = np.array(label_list)

data_list.clear()
label_list.clear()

print(y_test)

history = model.fit(x_train,y_train, batch_size = 32,
                    epochs = 15,
                    shuffle=True,
                    validation_data = (x_validate, y_validate))

print(f"\n SPLIT1 TESTING \n")

results = model.evaluate(x_test, y_test, batch_size = 32)
model.save_weights('MODEL1/')
