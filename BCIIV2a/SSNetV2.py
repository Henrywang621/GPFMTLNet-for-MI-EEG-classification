import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model



class gatemodule(Layer):

    def __init__(self, features_shape=48, **kwargs):
        super(gatemodule, self).__init__()
        self.shape = features_shape

    def build(self, input_shape):


        self.W1_f = self.add_weight(shape=(self.shape*2, self.shape),
                                   initializer='glorot_uniform',
                                   trainable=True,
                                   name='W1_f')
        
        self.b1_f = self.add_weight(shape=(self.shape,),
                                        initializer='zeros',
                                        trainable=True,
                                        name='b1_f')
        self.W2_f = self.add_weight(shape=(self.shape*2, self.shape),
                                   initializer='glorot_uniform',
                                   trainable=True,
                                   name='W2_f')
        
        self.b2_f = self.add_weight(shape=(self.shape,),
                                        initializer='zeros',
                                        trainable=True,
                                        name='b2_f')

    def call(self, inputs):
        input1, input2 = inputs
        concat = tf.concat([input1, input2], axis=-1)
        G1 = tf.nn.sigmoid(tf.matmul(concat, self.W1_f) + self.b1_f)
        G2 = tf.nn.sigmoid(tf.matmul(concat, self.W2_f) + self.b2_f)
        Gated_out1 = (1-G1)*input1+G1*input2
        Gated_out2 = (1-G2)*input2+G2*input1

        return Gated_out1, Gated_out2
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "shape": self.shape,
            "W1": self.W1_f.numpy(),
            "b1": self.b1_f.numpy(),
            "W2": self.W2_f.numpy(),
            "b2": self.b2_f.numpy(),
        })

        return config

        
        

def SSNetV2(trial_length = 960, nchans = 68, nclasses = 4):

    input_shape = (trial_length, nchans, 1)
    input1 = Input(shape=input_shape)
    input2 = Input(shape=input_shape)
    
    # Subnet for input construction
    Tin1 = Conv2D(48, (80, 1), padding='same', activation='relu')(input1)
    Tin1 = BatchNormalization()(Tin1)  
  

    # Subnet for EEG classification tasks
    TB1 = Conv2D(16, (16, 1), padding='same', activation='relu')(input2)
    TB1 = BatchNormalization()(TB1)
    TB2 = Conv2D(16, (32, 1), padding='same', activation='relu')(input2)
    TB2 = BatchNormalization()(TB2)
    TB3 = Conv2D(16, (64, 1), padding='same', activation='relu')(input2)
    TB3 = BatchNormalization()(TB3)
    TB_concat1 = tf.concat([TB1, TB2], axis = -1)
    TB_concat2 = tf.concat([TB_concat1, TB3], axis = -1)
    gate1_features1, gate1_features2 = gatemodule(features_shape=48)([TB_concat2, Tin1])

    TB_concat = AveragePooling2D((8,1))(gate1_features1)

    # Channel-wise attention
    TB_ca1 = GlobalAveragePooling2D()(TB_concat)
    TB_ca2 = Dense(32, activation="relu")(TB_ca1)
    TB_ca3 = Dense(48, activation="sigmoid")(TB_ca2)
    TB_out = Multiply()([TB_ca3, TB_concat])


    Tin2 = AveragePooling2D((4,1))(gate1_features2)
    # Tin2 = AveragePooling2D((8,1))(gate1_features2)
    Tin3 = Conv2D(48, (64, 1), padding='same', activation='relu')(Tin2)
    Tin3 = AveragePooling2D((2,1))(Tin3)


    gate2_features1, gate2_features2 = gatemodule(features_shape=48)([TB_out, Tin3])

    # Decoder4SS
    out1 = Conv2DTranspose(1, (trial_length-int(7/8*trial_length)+1, 1), activation='relu')(gate2_features2)
    out1 = Conv2DTranspose(1, (trial_length-int(3/4*trial_length)+1, 1), activation='relu')(out1)
    out1 = Conv2DTranspose(1, (trial_length-int(1/2*trial_length)+1, 1), activation='relu')(out1)
    out1 = Activation('relu', name='output1')(out1)  

    # Spatial filters
    EEG_r = Conv2D(32, (1, nchans), padding='valid', activation='relu')(gate2_features1)

    # Task 2 (EEG classification)
    out_c1 = Conv2D(16, (96, 1), padding='same', activation='relu')(EEG_r)
    out_c1 = BatchNormalization()(out_c1)  
    out_c2 = AveragePooling2D((4,1))(out_c1)  
    out_flatten = Flatten()(out_c2)
    out2  = Dense(nclasses, activation="softmax", name = 'output2')(out_flatten)   

    return Model(inputs = [input1, input2], outputs = [out1, out2]) 