# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This is the top-level file to train, evaluate or test your summarization model"""

import sys
import time
import os
import tensorflow as tf
import numpy as np
from collections import namedtuple
from data import Vocab
from batcher import Batcher
from model import SummarizationModel
from decode import BeamSearchDecoder
import util
from tensorflow.python import debug as tf_debug

FLAGS = tf.app.flags.FLAGS

# Where to find data
tf.app.flags.DEFINE_string('data_path', '/path/to/chunked/train_*', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('vocab_path', '/path/to/vocab', 'Path expression to text vocabulary file.')

# Important settings
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of train/eval/decode')
tf.app.flags.DEFINE_boolean('single_pass', False, 'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')

# Where to save output
tf.app.flags.DEFINE_string('log_root', '/Users/dfirebanks/Projects/DRLParaphrase/log/', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('exp_name', 'FirstTrial', 'Name for experiment. Logs will be saved in a directory with this name, under log_root.')

# Hyperparameters
tf.app.flags.DEFINE_integer('hidden_dim', 256, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')
tf.app.flags.DEFINE_integer('batch_size', 16, 'minibatch size')
tf.app.flags.DEFINE_integer('max_enc_steps', 400, 'max timesteps of encoder (max source text tokens)')
tf.app.flags.DEFINE_integer('max_dec_steps', 100, 'max timesteps of decoder (max summary tokens)')
tf.app.flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('min_dec_steps', 35, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
tf.app.flags.DEFINE_integer('vocab_size', 50000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
tf.app.flags.DEFINE_float('lr', 0.15, 'learning rate')
tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
tf.app.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')

# Pointer-generator or baseline model
tf.app.flags.DEFINE_boolean('pointer_gen', True, 'If True, use pointer-generator model. If False, use baseline model.')

# Coverage hyperparameters
tf.app.flags.DEFINE_boolean('coverage', False, 'Use coverage mechanism. Note, the experiments reported in the ACL paper train WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards. i.e. to reproduce the results in the ACL paper, turn this off for most of training then turn on for a short phase at the end.')
tf.app.flags.DEFINE_float('cov_loss_wt', 1.0, 'Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.')

# Utility flags, for restoring and changing checkpoints
tf.app.flags.DEFINE_boolean('convert_to_coverage_model', False, 'Convert a non-coverage model to a coverage model. Turn this on and run in train mode. Your current training model will be copied to a new version (same name with _cov_init appended) that will be ready to run with coverage flag turned on, for the coverage training stage.')
tf.app.flags.DEFINE_boolean('restore_best_model', False, 'Restore the best model in the eval/ dir and save it in the train/ dir, ready to be used for further training. Useful for early stopping, or if your training checkpoint has become corrupted with e.g. NaN values.')

# Debugging. See https://www.tensorflow.org/programmers_guide/debugger
tf.app.flags.DEFINE_boolean('debug', False, "Run in tensorflow's debug mode (watches for NaN/inf values)")
tf.app.flags.DEFINE_string('gpu', "6", "GPU to use")



def calc_running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.99):
    """Calculate the running average loss via exponential decay.
    This is used to implement early stopping w.r.t. a more smooth loss curve than the raw loss curve.

    Args:
        loss: loss on the most recent eval step
        running_avg_loss: running_avg_loss so far
        summary_writer: FileWriter object to write for tensorboard
        step: training iteration step
        decay: rate of exponential decay, a float between 0 and 1. Larger is smoother.

    Returns:
        running_avg_loss: new running average loss
    """
    if running_avg_loss == 0:    # on the first iteration just take the loss
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
    running_avg_loss = min(running_avg_loss, 12)    # clip
    loss_sum = tf.Summary()
    tag_name = 'running_avg_loss/decay=%f' % (decay)
    loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
    summary_writer.add_summary(loss_sum, step)
    tf.logging.info('running_avg_loss: %f', running_avg_loss)
    return running_avg_loss


def restore_best_model():
    """Load bestmodel file from eval directory, add variables for adagrad, and save to train directory"""
    tf.logging.info("Restoring bestmodel for training...")

    # Initialize all vars in the model
    sess = tf.Session(config=util.get_config())
    print("Initializing all variables...")
    sess.run(tf.initialize_all_variables())

    # Restore the best model from eval dir
    saver = tf.train.Saver([v for v in tf.all_variables() if "Adagrad" not in v.name])
    print("Restoring all non-adagrad variables from best model in eval dir...")
    curr_ckpt = util.load_ckpt(saver, sess, "eval")
    print("Restored %s." % curr_ckpt)

    # Save this model to train dir and quit
    new_model_name = curr_ckpt.split("/")[-1].replace("bestmodel", "model")
    new_fname = os.path.join(FLAGS.log_root, "train", new_model_name)
    print("Saving model to %s..." % (new_fname))
    new_saver = tf.train.Saver() # this saver saves all variables that now exist, including Adagrad variables
    new_saver.save(sess, new_fname)
    print("Saved.")
    exit()


def convert_to_coverage_model():
    """Load non-coverage checkpoint, add initialized extra variables for coverage, and save as new checkpoint"""
    tf.logging.info("converting non-coverage model to coverage model..")

    # initialize an entire coverage model from scratch
    sess = tf.Session(config=util.get_config())
    print( "initializing everything...")
    sess.run(tf.global_variables_initializer())

    # load all non-coverage weights from checkpoint
    saver = tf.train.Saver([v for v in tf.global_variables() if "coverage" not in v.name and "Adagrad" not in v.name])
    print( "restoring non-coverage variables...")
    curr_ckpt = util.load_ckpt(saver, sess)
    print( "restored.")

    # save this model and quit
    new_fname = curr_ckpt + '_cov_init'
    print( "saving model to %s..." % (new_fname))
    new_saver = tf.train.Saver() # this one will save all variables that now exist
    new_saver.save(sess, new_fname)
    print( "saved.")
    exit()

def cal_ranker_reward(batch, vocab, rank_word_dict, all_hyp, decoder, ranker, rank_sess, tg_len,
                      given_hyp=None):
    rewards = np.zeros((FLAGS.batch_size, FLAGS.max_dec_steps))
    rank_src = util.convert_to_full_vocab(batch.enc_batch_extend_vocab, rank_word_dict, vocab, batch.art_oovs)
    for idx in range(1, FLAGS.max_dec_steps):
        hyp = all_hyp.slice(idx)
        given_tokens = given_hyp[:, idx-1] if given_hyp is not None else None
        for _ in range(FLAGS.num_mc):
            sample_out, sample_out_extended_vocab = decoder.sample_batch_wise(
                batch, given_hyp=hyp, sampling=True, temp_ratio=FLAGS.sample_temp_ratio)
            sample_len = util.measure_len(sample_out, vocab)
            sample_len = np.array(sample_len) - 1
            rank_tg = util.convert_to_full_vocab(sample_out_extend_vocab, rank_word_dict, vocab, batch.art_oovs)
            rewards[:, idx-1] += ranker.eval(rank_sess, rank_src, rank_tg, batch.enc_lens, sample_len)
    return rewards

def setup_training(model, batcher):
    """Does setup before starting training (run_training)"""
    train_dir = os.path.join(FLAGS.log_root, "train")
    if not os.path.exists(train_dir): os.makedirs(train_dir)

    model.build_graph() # build the graph
    if FLAGS.convert_to_coverage_model:
        assert FLAGS.coverage, "To convert your non-coverage model to a coverage model, run with convert_to_coverage_model=True and coverage=True"
        convert_to_coverage_model()
    if FLAGS.restore_best_model:
        restore_best_model()
    saver = tf.train.Saver(max_to_keep=3) # keep 3 checkpoints at a time

    sv = tf.train.Supervisor(logdir=train_dir,
                                         is_chief=True,
                                         saver=saver,
                                         summary_op=None,
                                         save_summaries_secs=60, # save summaries for tensorboard every 60 secs
                                         save_model_secs=60, # checkpoint every 60 secs
                                         global_step=model.global_step)
    summary_writer = sv.summary_writer
    tf.logging.info("Preparing or waiting for session...")
    sess_context_manager = sv.prepare_or_wait_for_session(config=util.get_config())
    tf.logging.info("Created session.")
    try:
        run_training(model, batcher, sess_context_manager, sv, summary_writer) # this is an infinite loop until interrupted
    except KeyboardInterrupt:
        tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
        sv.stop()


def run_training(model, batcher, sess_context_manager, sv, summary_writer):
    """Repeatedly runs training iterations, logging loss to screen and writing summaries"""
    tf.logging.info("starting run_training")
    with sess_context_manager as sess:
        if FLAGS.debug: # start the tensorflow debugger
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        while True: # repeats until interrupted
            batch = batcher.next_batch()

            tf.logging.info('running training step...')
            t0=time.time()
            results = model.run_train_step(sess, batch)# np.ones((FLAGS.batch_size, FLAGS.max_dec_steps)))
            t1=time.time()
            tf.logging.info('seconds for training step: %.3f', t1-t0)

            loss = results['loss']
            tf.logging.info('loss: %f', loss) # print( the loss to screen

            if not np.isfinite(loss):
                raise Exception("Loss is not finite. Stopping.")


            summaries = results['summaries'] # we will write these summaries to tensorboard using summary_writer
            train_step = results['global_step'] # we need this to update our running average loss

            summary_writer.add_summary(summaries, train_step) # write the summaries
            if train_step % 100 == 0: # flush the summary writer every so often
                summary_writer.flush()


def run_eval(model, batcher, vocab):
    """Repeatedly runs eval iterations, logging to screen and writing summaries.
    Saves the model with the best loss seen so far."""
    model.build_graph() # build the graph
    saver = tf.train.Saver(max_to_keep=3) # we will keep 3 best checkpoints at a time
    sess = tf.Session(config=util.get_config())
    eval_dir = os.path.join(FLAGS.log_root, "eval") # make a subdir of the root dir for eval data
    bestmodel_save_path = os.path.join(eval_dir, 'bestmodel') # this is where checkpoints of best models are saved
    summary_writer = tf.summary.FileWriter(eval_dir)
    running_avg_loss = 0 # the eval job keeps a smoother, running average loss to tell it when to implement early stopping
    best_loss = None    # will hold the best loss achieved so far

    while True:
        _ = util.load_ckpt(saver, sess) # load a new checkpoint
        batch = batcher.next_batch() # get the next batch

        # run eval on the batch
        t0=time.time()
        results = model.run_eval_step(sess, batch)
        t1=time.time()
        tf.logging.info('seconds for batch: %.2f', t1-t0)

        # print( the loss and coverage loss to screen
        loss = results['loss']
        tf.logging.info('loss: %f', loss)
        if FLAGS.coverage:
            coverage_loss = results['coverage_loss']
            tf.logging.info("coverage_loss: %f", coverage_loss)

        # add summaries
        summaries = results['summaries']
        train_step = results['global_step']
        summary_writer.add_summary(summaries, train_step)

        # calculate running avg loss
        running_avg_loss = calc_running_avg_loss(np.asscalar(loss), running_avg_loss, summary_writer, train_step)

        # If running_avg_loss is best so far, save this checkpoint (early stopping).
        # These checkpoints will appear as bestmodel-<iteration_number> in the eval dir
        if best_loss is None or running_avg_loss < best_loss:
            tf.logging.info('Found new best model with %.3f running_avg_loss. Saving to %s', running_avg_loss, bestmodel_save_path)
            saver.save(sess, bestmodel_save_path, global_step=train_step, latest_filename='checkpoint_best')
            best_loss = running_avg_loss

        # flush the summary writer every so often
        if train_step % 100 == 0:
            summary_writer.flush()

def run_training_reinforce(hps, retrain_model_hps, eval_hps, model_train, model_decode, batcher, neg_batcher, vocab):
    train_dir = os.path.join(FLAGS.log_root, 'train')

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    config = util.get_config()
    default_device = tf.device('/cpu:0')

    with default_device():
        G_d = tf.Graph()
        G_t = tf.Graph()
        R = tf.Graph()

        with G_d.as_default():
            model_decode.build_graph()
            model_dvars = tf.get_collection(tf.GraphKeys.VARIABLES)

        with G_t.as_default():
            model_train.build_graph()
            model_tvars = tf.get_collection(tf.GraphKeys.VARIABLES)

        ranker, rank_sess, _ = MultiFeedForwardClassifier.load(
            FLAGS.rank_log_root, graph=R, batch_size=FLAGS.batch_size)

        rank_world_dict, rank_embeddings = ioutils.load_embeddings(
            FLAGS.rank_embed_size, FLAGS.rank_vocab_file,
            generate=False, load_extrac_from=FLAGS.rank_log_root, normalize=True)

        ranker.initialize_embeddings(rank_sess, rank_embeddings)

    decode_saver = tf.train.Saver(model_dvars)
    train_saver = tf.train.Saver(model_tvars, max_to_keep=10)
    model_ckpt_state = tf.train.get_checkpoint_state(train_dir)

    decode_sess = tf.Session(config=config, graph=G_d)
    train_sess = tf.Session(config=config, graph=G_t)
    train_saver = util.restore_model(train_sess, train_saver, model_ckpt_state.model_checkpoint_path, model_tvars, save=True)

    decode_saver = util.restore_model(decode_sess, decode_saver, model_ckpt_state.model_checkpoint_path, model_dvars)

    decoder = BeamSearchDecoder(model_decode, None, vocab, sess=decode_sess)

    train_step = 0

    hist_batch = []

    while True:
        batch = batcher.next_batch()
        neg_batch = neg_batcher.next_batch()
        pos_tg_len = np.sum(batch.padding_mask, axis=1)
        neg_tg_len = np.sum(neg_batch.padding_mask, axis=1)
        model_ckpt_state = train.train.get_checkpoint_state(train_dir)
        decode_saver.restore(decode_sess, model_ckpt_state.model_checkpoint_path)
        decoder = BeamSeachDecoder(model_decode, None, vocab, sess=decode_sess)
        pos_gen_out, pos_gen_out_extended_vocab, pos_all_hyp = decoder.sample_batch_wise(
            batch, sampling=FLAGS.sampling, preserve_hyp=True, temp_ratio=FLAGS.temp_ratio)
        neg_gen_out, neg_gen_out_extended_vocab, neg_all_hyp = decoder.sample_batch_wise(
            neg_batch, sampling=FLAGS.sampling, preserve_hyp=True, temp_ratio=FLAGS.temp_ratio)

        assert len(pos_gen_out) == hps.batch_size

        pos_decode_len = util.measure_len(pos_gen_out, vocab)
        neg_decode_len = util.measure_len(neg_gen_out, vocab)

        rank_src = util.convert_to_full_vocab(batch.enc_batch_extend_vocab, rank_word_dict, vocab, batch.art_oovs)
        rank_tg = util.convert_to_full_vocab(pos_gen_out_extended_vocab, rank_word_dict, vocab, batch.art_oovs)
        pos_sent_reward = ranker.eval(rank_sess, rank_src, rank_tg, batch.enc_lens, np.array(pos_decode_len)-1)

        rank_src = util.convert_to_full_vocab(
            neg_batch.enc_batch_extend_vocab, rank_word_dict, vocab, neg_batch.art_oovs)
        rank_tg = util.convert_to_full_vocab(
            neg_gen_out_extended_vocab, rank_word_dict, vocab, neg_batch.art_oovs)
        neg_sent_reward = ranker.eval(rank_sess, rank_src, rank_tg, neg_batch.enc_lens, np.array(neg_decode_len)-1)

        sent_reward = np.concatenate((pos_sent_reward, neg_sent_reward), axis=0)
        ###################################################################################
        # train G
        pos_decode_reward = cal_ranker_reward(
            batch, vocab, rank_word_dict, pos_all_hyp, decoder, ranker, rank_sess, pos_tg_len, rank_gt=FLAGS.rank_gt)
        neg_decode_reward = cal_ranker_reward(
            neg_batch, vocab, rank_word_dict, neg_all_hyp, decoder, ranker, rank_sess, neg_tg_len)
        decode_reward = np.concatenate((pos_decode_reward, neg_decode_reward), axis=0)

        msk = np.zeros_like(decode_reward)
        decode_len = np.concatenate((pos_decode_len, neg_decode_len), axis=0)

        for b, l in enumerate(decode_len):
            msk[b, :l] = 1

        rewards = decode_reward * msk / FLAGS.num_mc

        sent_reward = util.rank_sentence(sent_reward, scale=FLAGS.sent_scale)
        rewards = util.rescale_reward(rewards, sent_reward, msk, scale=FLAGS.token_scale)

        rewards = rewards * msk

        dec_inp, target, padding = util.prepare_retrain_data(
            hps, pos_gen_out, pos_gen_out_extended_vocab, vocab, pos_decode_len)
        neg_dec_inp, neg_target, neg_padding = util.prepare_retrain_data(
            hps, neg_gen_out, neg_gen_out_extended_vocab, vocab, neg_decode_len)

        dec_inp = np.concatenate((dec_inp, dec_inp, neg_dec_inp), axis=0)
        target = np.concatenate((target, target, neg_target), axis=0)
        padding = np.concatenate((padding, padding, neg_padding), axis=0)
        rewards = np.concatenate((
            rewards[:FLAGS.batch_size],
            np.ones(FLAGS.batch_size),
            rewards[FLAGS.batch_size:]
            ))
        rslts = model_train.run_train_step_with_reward(
            train_sess, batch, dec_inp, target, padding, rewards,
            neg_batch=neg_batch, temp_ratio=FLAGS.temp_ratio)

        train_step = rslts['global_step']
        train_saver.save(
            train_sess,
            os.path.join(train_dir, 'model.ckpt'),
            global_step=train_step
            )

        if train_step & 50 == 0:
            train_saver.save(
                train_sess, os.path.join(save_dir, 'model.ckpt'),
                global_step=train_step
                )

            eval_loss = 0
            eval_batcher = SimpleBatcher(FLAGS.eval_data_path, vocab, eval_hps)
            eval_batcher_sz = SimpleBatcher(FLAGS.eval_data_path, vocab, eval_hps)

            eval_saver.restore(eval_sess, model_ckpt_state.model_checkpoint_path)
            for _ in range(eval_batcher.num_batch):
                eval_batch = eval_batcher.next_batch()
                eval_loss += model_eval.run_eval_step(eval_sess, eval_batch)['loss']

            score = 0
            sample_score = 0
            rouge_score = 0
            rouge2_score = 0
            rougel_score = 0
            bleu_score = 0

            for j in range(eval_batcher_sz.num_batch):
                eval_batch = eval_batcher_sz.next_batch()
                tg_len = np.sum(eval_batch.padding_mask, axis=1)
                _, sample_out_extended_vocab = decoder.sample_batch_wise(
                    eval_batch, sampling=FLAGS.rank_sampling, temp_ratio=FLAGS.sample_temp_ratio)
                sample_len = util.measure_len(sample_out_extended_vocab, vocab)
                gen_out, gen_out_extended_vocab = decoder.sample_batch_wise

def main(unused_argv):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    if len(unused_argv) != 1: # print(s a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    tf.logging.set_verbosity(tf.logging.INFO) # choose what level of logging you want
    tf.logging.info('Starting seq2seq_attention in %s mode...', (FLAGS.mode))

    # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
    FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)
    if not os.path.exists(FLAGS.log_root):
        if FLAGS.mode=="train":
            os.makedirs(FLAGS.log_root)
        else:
            raise Exception("Logdir %s doesn't exist. Run in train mode to create it." % (FLAGS.log_root))

    vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size) # create a vocabulary

    # If in decode mode, set batch_size = beam_size
    # Reason: in decode mode, we decode one example at a time.
    # On each step, we have beam_size-many hypotheses in the beam, so we need to make a batch of these hypotheses.
    if FLAGS.mode == 'decode':
        FLAGS.batch_size = FLAGS.beam_size

    # If single_pass=True, check we're in decode mode
    if FLAGS.single_pass and FLAGS.mode!='decode':
        raise Exception("The single_pass flag should only be True in decode mode")

    # Make a namedtuple hps, containing the values of the hyperparameters that the model needs
    hparam_list = ['mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm', 'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps', 'max_enc_steps', 'coverage', 'cov_loss_wt', 'pointer_gen']
    hps_dict = {}
    for key,val in FLAGS.flag_values_dict().items(): # for each flag
        if key in hparam_list: # if it's in the list
            hps_dict[key] = val # add it to the dict
    hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)
    print(hps.mode)
    print(hps.batch_size)

    # Create a batcher object that will create minibatches of data
    batcher = Batcher(FLAGS.data_path, vocab, hps, single_pass=FLAGS.single_pass)

    tf.set_random_seed(111) # a seed value for randomness

    if hps.mode == 'train':
        print( "creating model...")
        model = SummarizationModel(hps, vocab)
        setup_training(model, batcher)
    elif hps.mode == 'eval':
        model = SummarizationModel(hps, vocab)
        run_eval(model, batcher, vocab)
    elif hps.mode == 'decode':
        decode_model_hps = hps    # This will be the hyperparameters for the decoder model
        decode_model_hps = hps._replace(max_dec_steps=1) # The model is configured with max_dec_steps=1 because we only ever run one step of the decoder at a time (to do beam search). Note that the batcher is initialized with max_dec_steps equal to e.g. 100 because the batches need to contain the full summaries
        model = SummarizationModel(decode_model_hps, vocab)
        decoder = BeamSearchDecoder(model, batcher, vocab)
        decoder.decode() # decode indefinitely (unless single_pass=True, in which case deocde the dataset exactly once)
    else:
        raise ValueError("The 'mode' flag must be one of train/eval/decode")

if __name__ == '__main__':
    tf.app.run()
