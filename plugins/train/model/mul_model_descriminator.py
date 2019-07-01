#!/usr/bin/env python3
""" Original Model
    Based on the original https://www.reddit.com/r/deepfakes/
    code sample + contribs """

from keras.layers import Conv2D, Dense, Flatten, Input, Reshape

from keras.losses import binary_crossentropy , mean_absolute_error

from keras.models import Model as KerasModel

from .mul_model_base import ModelBase, logger

import tensorflow as tf 


class Model(ModelBase):
    """ mul model descriminators Model """
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)

        self.configfile = kwargs.get("configfile", None)
        if "input_shape" not in kwargs:
            kwargs["input_shape"] = (64, 64, 3)
        if "encoder_dim" not in kwargs:
            kwargs["encoder_dim"] = 512 if self.config["lowmem"] else 1024
        logger.debug("Initialized %s", self.__class__.__name__)
        self.discriminators = dict() # dict to store discriminators
        self.gans = dict() # dict stores gan network
        super().__init__(*args, **kwargs)
        self.trainer = "mul_model_descriminator_trainer"
        

    #not compiling predictors
    def build(self):
        """ Build the model. Override for custom build methods """
        self.add_networks()
        #self.print_networks("discriminator")
        self.load_models(swapped=False)
        self.build_autoencoders()
        self.build_discriminators()
        self.build_gans()
        #self.unfreezeDiscriminators()
        self.log_summary()
        #self.compile_predictors(initialize=True)
        self.compile_gans(initialize=True)
        #self.print_gan()
        self.compile_discriminators()

    def print_networks(self,model):
        for i in range(self.num_of_sides):
            print(self.networks[f"{model}_{i}"].network.summary())

    def print_gan(self):
        for gan in self.gans:
            print(self.gans[gan].summary())

    def print_discriminators(self):
        for i in self.discriminators:
            print(self.discriminators[i].summary())

    def freezeModel(self,model):
        for layer in model.layers:
            layer.trainable = False

    def unfreezeModel(self,model):
        for layer in model.layers:
            layer.trainable = True

    def freezeDiscriminators(self):
        for i in self.discriminators:
            self.freezeModel(self.discriminators[i])

    def unfreezeDiscriminators(self):
        for i in self.discriminators:
            self.unfreezeModel(self.discriminators[i])

    def add_networks(self):
        """ Add the original model weights """
        logger.debug("Adding networks")
        #adding decoders
        for i in range(self.num_of_sides):
            self.add_network("decoder", str(i), self.decoder())
        #adding encoder
        self.add_network("encoder", None, self.encoder())
        #adding discriminator
        for i in range(self.num_of_sides):
            self.add_network("discriminator", str(i) , self.discriminator())
        logger.debug("Added networks")

    def add_discriminator(self,side,model):
        """ Add a predictor to the predictors dictionary """
        logger.debug("Adding discriminator: (side: '%s', model: %s)", side, model)
        if self.gpus > 1:
            logger.debug("Converting to multi-gpu: side %s", side)
            model = multi_gpu_model(model, self.gpus)
        self.discriminators[side] = model

    def add_gan(self,side,model):
        """ Add a predictor to the predictors dictionary """
        logger.debug("Adding discriminator: (side: '%s', model: %s)", side, model)
        if self.gpus > 1:
            logger.debug("Converting to multi-gpu: side %s", side)
            model = multi_gpu_model(model, self.gpus)
        self.gans[side] = model

    def build_discriminators(self):
        for side in range(self.num_of_sides):
            self.add_discriminator(str(side), self.networks[f"discriminator_{side}"].network)
        logger.debug("Discriminator model")

    def build_gans(self):
        inputs = [Input(shape = self.input_shape , name = "face")]
        for side in range(self.num_of_sides):
            outputs_list = []
            encoded_output = self.networks["encoder"].network(inputs[0])
            for side1 in range(self.num_of_sides):
                if (side1 == side):
                    reconstructed_output = self.networks["decoder_{}".format(side)].network(encoded_output)
                    outputs_list.insert(0,reconstructed_output)
                else:
                    swaped_output = self.networks[f"decoder_{side1}"].network(encoded_output)
                    discriminator_output = self.networks[f"discriminator_{side1}"].network(swaped_output)
                    outputs_list.append(discriminator_output)
            self.add_gan(str(side) , KerasModel(inputs , outputs_list , name = "gan") )

    def compile_gans(self, initialize=True):
        """ Compile the predictors """
        logger.debug("Compiling gans")
        learning_rate = self.config.get("learning_rate", 5e-5)
        optimizer = self.get_optimizer(lr=learning_rate, beta_1=0.5, beta_2=0.999)

        for side, model in self.gans.items():
            loss_names = ["total_loss"]
            loss_func = [self.loss_function(None , side , initialize)]
            for i in range(1,self.num_of_sides):
                loss_func.append(binary_crossentropy)
            model.compile(optimizer=optimizer, loss=loss_func)
            if len(loss_names) > 1:
                loss_names.insert(0, "total_loss")
            if initialize:
                self.state.add_session_loss_names(f"gan_{side}", loss_names)
                self.history[f"gan_{side}"] = list()
        logger.debug("Compiled gans. Losses: %s", loss_names)

    def compile_discriminators(self, initialize = True):
        """ Compile the predictors """
        logger.debug("Compiling discriminators")
        learning_rate = self.config.get("learning_rate", 5e-5)
        optimizer = self.get_optimizer(lr=learning_rate, beta_1=0.5, beta_2=0.999)

        for side, model in self.discriminators.items():
            loss_names = ["loss_discriminator"]
            model.trainable = True
            model.compile(optimizer=optimizer , loss = binary_crossentropy)

            if len(loss_names) > 1:
                loss_names.insert(0, "total_loss")
            if initialize:
                self.state.add_session_loss_names(f"discriminator_{side}", loss_names)
                self.history[f"discriminator_{side}"] = list()
        logger.debug("Compiled gans. Losses: %s", loss_names)

    def build_autoencoders(self):
        """ Initialize original model """
        logger.debug("Initializing model")
        inputs = [Input(shape=self.input_shape, name="face")]
        if self.config.get("mask_type", None):
            mask_shape = (self.input_shape[:2] + (1, ))
            inputs.append(Input(shape=mask_shape, name="mask"))

        for side in range(self.num_of_sides):
            logger.debug("Adding Autoencoder. Side: %s", str(side))
            decoder = self.networks["decoder_{}".format(side)].network
            output = decoder(self.networks["encoder"].network(inputs[0]))
            autoencoder = KerasModel(inputs, output)
            self.add_predictor(str(side), autoencoder)
        logger.debug("Initialized model")

    def encoder(self):
        """ Encoder Network """
        input_ = Input(shape=self.input_shape)
        var_x = input_
        var_x = self.blocks.conv(var_x, 128)
        var_x = self.blocks.conv(var_x, 256)
        var_x = self.blocks.conv(var_x, 512)
        if not self.config.get("lowmem", False):
            var_x = self.blocks.conv(var_x, 1024)
        var_x = Dense(self.encoder_dim)(Flatten()(var_x))
        var_x = Dense(4 * 4 * 1024)(var_x)
        var_x = Reshape((4, 4, 1024))(var_x)
        var_x = self.blocks.upscale(var_x, 512)
        return KerasModel(input_, var_x)

    def discriminator(self):
        """ Descriminator Network """
        input_ = Input(shape = self.input_shape , name = "discriminator_input" )
        varx = input_
        varx = self.blocks.conv(varx , 128)
        varx = self.blocks.conv(varx , 256)
        varx = Dense(1)(Flatten()(varx))
        return KerasModel(input_ , varx)

    def decoder(self):
        """ Decoder Network """
        input_ = Input(shape=(8, 8, 512))
        var_x = input_
        var_x = self.blocks.upscale(var_x, 256)
        var_x = self.blocks.upscale(var_x, 128)
        var_x = self.blocks.upscale(var_x, 64)
        var_x = Conv2D(3, kernel_size=5, padding="same", activation="sigmoid")(var_x)
        outputs = [var_x]

        if self.config.get("mask_type", None):
            var_y = input_
            var_y = self.blocks.upscale(var_y, 256)
            var_y = self.blocks.upscale(var_y, 128)
            var_y = self.blocks.upscale(var_y, 64)
            var_y = Conv2D(1, kernel_size=5, padding='same', activation='sigmoid')(var_y)
            outputs.append(var_y)
        return KerasModel(input_, outputs=outputs)