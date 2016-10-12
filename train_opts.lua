local M = { }

function M.parse(arg)

  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Train an Image Captioning model')
  cmd:text()
  cmd:text('Options')

  -- Data input settings
  cmd:option('-checkpoint_start_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')

  cmd:option('-dataset', 'referit', 'The dataset to use')
  cmd:option('-nThreads', 8, 'Number of threads')
  
  -- Model settings
  cmd:option('-rnn_size',512,'size of the rnn in number of hidden nodes in each layer')
  cmd:option('-image_l1_size',1024,'the encoding size of each token in the vocabulary, and the image.')
  cmd:option('-image_l2_size',1024,'the encoding size of each token in the vocabulary, and the image.')
  cmd:option('-score_type','dot','dot, concat, euclidean, cosine, cbp')
  cmd:option('-loss_type','structure','structure, softmax, softmax2, logsitic, hinge.')
  cmd:option('-visnet_type','old','rt, lc, old.')
  cmd:option('-state_type', 'final', 'avg, final')
  cmd:option('-drop_prob_vis', 0., 'strength of dropout in the visnet')
  cmd:option('-slack_rescaled',0,'Rescaled slack or not.')
  cmd:option('-normalize',0,'Normalize the feature or not.')
  cmd:option('-margin',1,'The margin of hinge loss.')


  -- Optimization: General
  cmd:option('-max_iters', -1, 'max number of iterations to run for (-1 = run forever)')
  --cmd:option('-batch_size',16,'what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')
  cmd:option('-grad_clip',0.1,'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
  cmd:option('-drop_prob_lm', 0.5, 'strength of dropout in the Language Model RNN')

  -- Optimization: for the Language Model
  cmd:option('-optim','adam','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
  cmd:option('-learning_rate',4e-4,'learning rate')
  cmd:option('-learning_rate_decay_start', -1, 'at what iteration to start decaying learning rate? (-1 = dont)')
  cmd:option('-learning_rate_decay_every', 50000, 'every how many iterations thereafter to drop LR by half?')
  cmd:option('-optim_alpha',0.8,'alpha for adagrad/rmsprop/momentum/adam')
  cmd:option('-optim_beta',0.999,'beta used for adam')
  cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')

  -- Evaluation/Checkpointing
  cmd:option('-val_images_use', 3200, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
  cmd:option('-save_checkpoint_every', 2500, 'how often to save a model checkpoint?')
  cmd:option('-checkpoint_path', '', 'folder to save checkpoints into (empty = this folder)')
  cmd:option('-losses_log_every', 25, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
  cmd:option('-load_best_score', 0, 'Load best score or not')


  -- misc
  cmd:option('-backend', 'cudnn', 'nn|cudnn')
  cmd:option('-id', '', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
  cmd:option('-seed', 123, 'random number generator seed to use')
  cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')

  cmd:text()

  -------------------------------------------------------------------------------
  -- Basic Torch initializations
  -------------------------------------------------------------------------------
  local opt = cmd:parse(arg)
  return opt
end

return M