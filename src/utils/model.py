import tensorflow as tf

def create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NO_CLASSES):

    LAYERS = [
          
          tf.keras.layers.Flatten(input_shape=[28,28], name='input_layer'),
          tf.keras.layers.Dense(300, activation = 'relu', name='hidden_layer1'),
          tf.keras.layers.Dense(100, activation='relu', name='hiddenlayer2'),
          tf.keras.layers.Dense(10, activation='softmax', name='output_layer')

            ]

    model_classifier = tf.keras.models.Sequential(LAYERS)

    model_classifier.summary()

    model_classifier.compile(loss=LOSS_FUNCTION, optimizer = OPTIMIZER, metrics = METRICS)

    return model_classifier