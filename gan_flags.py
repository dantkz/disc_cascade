
# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_STEPS_PER_DECAY = 3800.0      # steps after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
GEN_INITIAL_LEARNING_RATE = 0.0001      # Initial learning rate.
DISC_INITIAL_LEARNING_RATE = 0.0001      # Initial learning rate.

log_device_placement = False
eval_interval_secs = 60 * 1

train_dir = 'gan_logs'

