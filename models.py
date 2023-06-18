#!/usr/bin/python3

from enum import Enum;
import numpy as np;
import tensorflow as tf;
import tensorflow_addons as tfa;

def TimestepEmbedding(time_embedding_channels):
  inputs = tf.keras.Input((), dtype = tf.int32); # inputs.shape = (batch)
  half_dim = tf.keras.layers.Lambda(lambda x, d: tf.constant(d, dtype = tf.int32) // 2, arguments = {'d': time_embedding_channels})(inputs); # half_dim.shape = (,)
  emb = tf.keras.layers.Lambda(lambda x: tf.math.log(10000.) / (tf.cast(x, dtype = tf.float32) - 1))(half_dim); # emb.shape = (,)
  emb = tf.keras.layers.Lambda(lambda x: tf.exp(tf.range(x[0], dtype = tf.float32) * -x[1]))([half_dim, emb]); # emb.shape = (dim/2,)
  emb = tf.keras.layers.Lambda(lambda x: tf.expand_dims(tf.cast(x[0], dtype = tf.float32), axis = -1) * tf.expand_dims(x[1], axis = 0))([inputs, emb]) # emb.shape = (batch, dim/2)
  emb = tf.keras.layers.Lambda(lambda x: tf.concat([tf.math.sin(x), tf.math.cos(x)], axis = 1))(emb); # emb.shape = (batch, dim)
  if time_embedding_channels % 2 == 1:
    emb = tf.keras.layers.Lambda(lambda x: tf.pad(x, ((0,0),(0,1))))(emb);
  return tf.keras.Model(inputs = inputs, outputs = emb);

def ResnetBlock(time_embedding_channels, input_channels, output_channels = None, conv_shortcut = False, drop_rate = 0.):
  if output_channels is None: output_channels = input_channels;
  inputs = tf.keras.Input((None, None, input_channels)); # inputs.shape = (batch, h, w, input_channels)
  time_embeds = tf.keras.Input((time_embedding_channels,)); # time_embeds.shape = (batch, embed_dim)
  results = tfa.layers.GroupNormalization()(inputs); # results.shape = (batch, h, w, input_channels)
  results = tf.keras.layers.Lambda(lambda x: tf.keras.activations.swish(x))(results); # results.shape = (batch, h, w, input_channels)
  results = tf.keras.layers.Conv2D(output_channels, kernel_size = (3,3), padding = 'same', kernel_initializer = tf.keras.initializers.VarianceScaling(scale = 1.0, mode = 'fan_avg', distribution = 'uniform'), bias_initializer = tf.keras.initializers.Zeros())(results); # results.shape = (b, h, w, output_channels)
  results_time = tf.keras.layers.Lambda(lambda x: tf.keras.activations.swish(x))(time_embeds); # results_time_embed.shape = (batch, embed_dim)
  results_time = tf.keras.layers.Dense(output_channels, kernel_initializer = tf.keras.initializers.VarianceScaling(scale = 1.0, mode = 'fan_avg', distribution = 'uniform'), bias_initializer = tf.keras.initializers.Zeros())(results_time);
  results_time = tf.keras.layers.Reshape((1, 1, output_channels))(results_time); # results_time.shape = (batch, 1, 1, embed_dim)
  results = tf.keras.layers.Add()([results, results_time]); # results.shape = (batch, h, w, output_channels + embed_dim)
  results = tfa.layers.GroupNormalization()(results); # results.shape = (batch, h, w, output_channels + embed_dim)
  results = tf.keras.layers.Lambda(lambda x: tf.keras.activations.swish(x))(results); # results.shape = (batch, h, w, output_channels + embed_dim)
  results = tf.keras.layers.Dropout(rate = drop_rate)(results); # results.shape = (batch, h, w, output_channels + embed_dim)
  results = tf.keras.layers.Conv2D(output_channels, kernel_size = (3,3), padding = 'same', kernel_initializer = tf.keras.initializers.VarianceScaling(scale = 1e-10, mode = 'fan_avg', distribution = 'uniform'), bias_initializer = tf.keras.initializers.Zeros())(results); # results.shape = (batch, h, w, output_channels)
  if input_channels != output_channels:
    if conv_shortcut:
      residual = tf.keras.layers.Conv2D(output_channels, kernel_size = (3,3), padding = 'same', kernel_initializer = tf.keras.initializers.VarianceScaling(scale = 1.0, mode = 'fan_avg', distribution = 'uniform'), bias_initializer = tf.keras.initializers.Zeros())(inputs);
    else:
      residual = tf.keras.layers.Dense(output_channels, kernel_initializer = tf.keras.initializers.VarianceScaling(scale = 1.0, mode = 'fan_avg', distribution = 'uniform'), bias_initializer = tf.keras.initializers.Zeros())(inputs);
  else:
    residual = tf.keras.layers.Identity()(inputs);
  results = tf.keras.layers.Add()([results, residual]);
  return tf.keras.Model(inputs = (inputs, time_embeds), outputs = results);

def AttentionBlock(input_channels):
  inputs = tf.keras.Input((None, None, input_channels));
  results = tfa.layers.GroupNormalization()(inputs);
  q = tf.keras.layers.Dense(input_channels, kernel_initializer = tf.keras.initializers.VarianceScaling(scale = 1.0, mode = 'fan_avg', distribution = 'uniform'), bias_initializer = tf.keras.initializers.Zeros())(results);
  k = tf.keras.layers.Dense(input_channels, kernel_initializer = tf.keras.initializers.VarianceScaling(scale = 1.0, mode = 'fan_avg', distribution = 'uniform'), bias_initializer = tf.keras.initializers.Zeros())(results);
  v = tf.keras.layers.Dense(input_channels, kernel_initializer = tf.keras.initializers.VarianceScaling(scale = 1.0, mode = 'fan_avg', distribution = 'uniform'), bias_initializer = tf.keras.initializers.Zeros())(results);
  w = tf.keras.layers.Lambda(lambda x, c: tf.reshape(tf.einsum('bhwc,bHWc->bhwHW', x[0], x[1]) * (tf.cast(c, dtype = tf.float32) ** (-0.5)), (tf.shape(x[0])[0], tf.shape(x[0])[1], tf.shape(x[0])[2], tf.shape(x[1])[1] * tf.shape(x[1])[2])), arguments = {'c': input_channels})([q, k]);
  w = tf.keras.layers.Softmax(axis = -1)(w);
  w = tf.keras.layers.Lambda(lambda x: tf.reshape(x[0], (tf.shape(x[0])[0], tf.shape(x[0])[1], tf.shape(x[0])[2], tf.shape(x[1])[1], tf.shape(x[1])[2])))([w, k]);
  h = tf.keras.layers.Lambda(lambda x: tf.einsum('bhwHW, bHWc->bhwc', x[0], x[1]))([w, v]);
  h = tf.keras.layers.Dense(input_channels, kernel_initializer = tf.keras.initializers.VarianceScaling(scale = 1e-10, mode = 'fan_avg', distribution = 'uniform'), bias_initializer = tf.keras.initializers.Zeros())(h);
  results = tf.keras.layers.Add()([inputs, h]);
  return tf.keras.Model(inputs = inputs, outputs = results);

def Unet(input_shape, input_channels, output_channels, res_block_num, attn_resolutions = [8,], channel_multipliers = (1, 2, 4, 8), drop_rate = 0., resample_with_conv = True):
  inputs = tf.keras.Input(input_shape); # inputs.shape = (batch, w, w, c)
  t = tf.keras.Input((), dtype = tf.int32); # t.shape = (batch)
  time_embed = TimestepEmbedding(input_channels)(t); # time_embed.shape = (batch, input_channels)
  time_embed = tf.keras.layers.Dense(4 * input_channels, activation = tf.keras.activations.swish, kernel_initializer = tf.keras.initializers.VarianceScaling(scale = 1.0, mode = 'fan_avg', distribution = 'uniform'), bias_initializer = tf.keras.initializers.Zeros())(time_embed); # time_embed.shape = (batch, 4 * input_channels)
  time_embed = tf.keras.layers.Dense(4 * input_channels, kernel_initializer = tf.keras.initializers.VarianceScaling(scale = 1.0, mode = 'fan_avg', distribution = 'uniform'), bias_initializer = tf.keras.initializers.Zeros())(time_embed); # time_embed.shape = (batch, 4 * input_channels)
  # downsampling
  downsampling_results = list();
  results = tf.keras.layers.Conv2D(input_channels, kernel_size = (3,3), padding = 'same', kernel_initializer = tf.keras.initializers.VarianceScaling(scale = 1.0, mode = 'fan_avg', distribution = 'uniform'), bias_initializer = tf.keras.initializers.Zeros())(inputs);
  downsampling_results.append(results);
  for i_level, level_multiplier in enumerate(channel_multipliers):
    # residual blocks for this resolution
    for i_block in range(res_block_num):
      results = ResnetBlock(input_channels * 4, results.shape[-1], input_channels * level_multiplier, drop_rate = drop_rate)([results, time_embed]);
      if results.shape[1] in attn_resolutions:
        results = AttentionBlock(results.shape[-1])(results);
      downsampling_results.append(results);
    # downsample
    if i_level != len(channel_multipliers) - 1:
      if resample_with_conv:
        results = tf.keras.layers.Conv2D(results.shape[-1], kernel_size = (3,3), strides = (2,2), padding = 'same', kernel_initializer = tf.keras.initializers.VarianceScaling(scale = 1.0, mode = 'fan_avg', distribution = 'uniform'), bias_initializer = tf.keras.initializers.Zeros())(results);
      else:
        results = tf.keras.layers.AveragePooling2D(pool_size = (2,2), padding = 'same')(results);
      downsampling_results.append(results);
  # middle
  results = ResnetBlock(input_channels * 4, results.shape[-1], results.shape[-1], drop_rate = drop_rate)([results, time_embed]);
  results = AttentionBlock(results.shape[-1])(results);
  results = ResnetBlock(input_channels * 4, results.shape[-1], results.shape[-1], drop_rate = drop_rate)([results, time_embed]);
  # upsampling
  for i_level in reversed(range(len(channel_multipliers))):
    level_multiplier = channel_multipliers[i_level];
    # residual blocks for this resolution
    for i_block in range(res_block_num + 1):
      results = tf.keras.layers.Concatenate(axis = -1)([results, downsampling_results.pop()]);
      results = ResnetBlock(input_channels * 4, results.shape[-1], input_channels * level_multiplier, drop_rate = drop_rate)([results, time_embed]);
      if results.shape[1] in attn_resolutions:
        results = AttentionBlock(results.shape[-1])(results);
    # upsample
    if i_level != 0:
      results = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, size = (tf.shape(x)[1] * 2, tf.shape(x)[2] * 2), method = tf.image.ResizeMethod.NEAREST_NEIGHBOR))(results);
      if resample_with_conv:
        results = tf.keras.layers.Conv2D(results.shape[-1], kernel_size = (3,3), padding = 'same', kernel_initializer = tf.keras.initializers.VarianceScaling(scale = 1.0, mode = 'fan_avg', distribution = 'uniform'), bias_initializer = tf.keras.initializers.Zeros())(results);
  # end
  results = tfa.layers.GroupNormalization()(results);
  results = tf.keras.layers.Lambda(lambda x: tf.keras.activations.swish(x))(results);
  results = tf.keras.layers.Conv2D(output_channels, kernel_size = (3,3), padding = 'same', kernel_initializer = tf.keras.initializers.VarianceScaling(scale = 1e-10, mode = 'fan_avg', distribution = 'uniform'), bias_initializer = tf.keras.initializers.Zeros())(results);
  return tf.keras.Model(inputs = (inputs, t), outputs = results);

class BetaSchedule(Enum):
  quad = 1
  linear = 2
  warmup10 = 3
  warmup50 = 4
  const = 5
  jsd = 6

def get_beta_schedule(scheduler: BetaSchedule, start, end, timesteps):
  def _warmup_beta(start, end, timesteps, frac):
    betas = end * np.ones(timesteps, dtype = np.float64)
    warmup_time = int(timesteps * frac)
    betas[:warmup_time] = np.linspace(start, end, warmup_time, dtype = np.float64)
    return betas
  if scheduler == BetaSchedule.quad:
    betas = np.linspace(start ** 0.5, end ** 0.5, timesteps, dtype = np.float64) ** 2;
  elif scheduler == BetaSchedule.linear:
    betas = np.linspace(start, end, timesteps, dtype = np.float64);
  elif scheduler == BetaSchedule.warmup10:
    betas = _warmup_beta(start, end, timesteps, 0.1);
  elif scheduler == BetaSchedule.warmup50:
    betas = _warmup_beta(start, end, timesteps, 0.5);
  elif scheduler == BetaSchedule.const:
    betas = end * np.ones(timesteps, dtype = np.float64);
  elif scheduler == BetaSchedule.jsd:
    betas = 1. / np.linspace(timesteps, 1, timesteps, dtype = np.float64);
  else:
    raise NotImplementedError(scheduler)
  return betas

if __name__ == "__main__":
  import numpy as np
  inputs = np.random.randint(low = 0, high = 10, size = (20));
  emb = TimestepEmbedding(200)(inputs)
  print(emb.shape)
  inputs = np.random.normal(size = (4, 32, 32, 128))
  emb = np.random.normal(size = (4, 256))
  results = ResnetBlock(256, 128, 512)([inputs, emb])
  print(results.shape)
  results = AttentionBlock(128)(inputs);
  print(results.shape)
  inputs = np.random.normal(size = (4,64,64,8));
  t = np.random.randint(low = 0, high = 10, size = (4,))
  unet = Unet((64,64,8), 128, 256, 2, (16,))
  results = unet([inputs,t]);
  unet.save('unet.h5')
  print(results.shape)
