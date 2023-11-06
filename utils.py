"""
Based on work by: Pi-Yueh Chuang <pychuang@gwu.edu>
URL: https://gist.github.com/piyueh/712ec7d4540489aad2dcfb80f9a54993#file-tf_keras_tfp_lbfgs-py
"""
import tensorflow as tf
import numpy

def function_factory(model, loss, X, X_0, X_1, X_2, X_3, X_4):
    """A factory to create a function required by tfp.optimizer.lbfgs_minimize.

    Args:
        model [in]: an instance of `tf.keras.Model` or its subclasses.
        loss [in]: a function with signature loss_value = loss(pred_y, true_y).
        train_x [in]: the input part of training data.
        train_y [in]: the output part of training data.

    Returns:
        A function that has a signature of:
            loss_value, gradients = f(model_parameters).
    """

    # obtain the shapes of all trainable parameters in the model
    # shapes = tf.shape_n(model.trainable_variables)
    # Get layer shapes
    # shapes = tf.shape_n(model.trainable_variables) # This is not working for me. I use the following instead
    shapes = []
    for i in range(len(model.trainable_variables)):
        shapes.append(model.trainable_variables[i].shape)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = [] # stitch indices
    part = [] # partition indices

    for i, shape in enumerate(shapes):
        n = numpy.product(shape)
        idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n

    part = tf.constant(part)

    @tf.function
    def assign_new_model_parameters(params_1d):
        """A function updating the model's parameters with a 1D tf.Tensor.

        Args:
            params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
        """

        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model.trainable_variables[i].assign(tf.reshape(param, shape))

    # now create a function that will be returned by this factory
    @tf.function
    def f(params_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.

        This function is created by function_factory.

        Args:
           params_1d [in]: a 1D tf.Tensor.

        Returns:
            A scalar loss and the gradients w.r.t. the `params_1d`.
        """

        # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
        with tf.GradientTape() as tape:
            # update the parameters in the model
            assign_new_model_parameters(params_1d)
            # calculate the loss
            # loss_value = loss(model(train_x, training=True), train_y)
            loss_value, loss_m, loss_b, loss_i = loss(model, X, X_0, X_1, X_2, X_3, X_4)

        # calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(loss_value, model.trainable_variables)
        grads = tf.dynamic_stitch(idx, grads)

        # print out iteration & loss
        f.iter.assign_add(1)
        if f.iter % 100 == 0:
            tf.print("i:", f.iter, "Total loss:", loss_value, 'PDE loss:', loss_m, 'BC loss:', loss_b, 'IC loss:', loss_i)

        # store loss value so we can retrieve later
        tf.py_function(f.history.append, inp=[[loss_value, loss_m, loss_b, loss_i]], Tout=[])

        return loss_value, grads

    # store these information as members so we can use them outside the scope
    f.iter = tf.Variable(0)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters
    f.history = []

    return f