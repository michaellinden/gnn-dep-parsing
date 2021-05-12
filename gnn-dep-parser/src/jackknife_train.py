from antu.utils.dual_channel_logger import dual_channel_logger
from utils.conllu_reader import PTBReader
from utils.PTB_dataset import DatasetSetting, PTBDataset
from antu.io.ext_embedding_readers import glove_reader
from antu.io import Vocabulary
from antu.io.configurators import IniConfigurator
from collections import Counter
import numpy as np
import argparse
import _pickle
import math
import os
import random
import sys
import time
import logging
import multiprocessing as mp
random.seed(666)
np.random.seed(666)

#from antu.nn.dynet.attention.multi_head import MultiHeadedAttention, MultiLayerMultiHeadAttention
from antu.nn.dynet.attention.my_attention import ScaledDotProductAttention, MyMultiHeadAttention, LabelAttention, Encoder



def main():
    # Configuration file processing
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='../configs/debug.cfg')
    argparser.add_argument('--continue_training', action='store_true',
                           help='Load model Continue Training')
    argparser.add_argument('--name', default='experiment',
                           help='The name of the experiment.')
    argparser.add_argument('--model', default='s2s',
                           help='s2s: seq2seq-head-selection-model'
                           's2tDFS: seq2tree-DFS-decoder-model')
    argparser.add_argument('--gpu', default='0', help='GPU ID (-1 to cpu)')
    args, extra_args = argparser.parse_known_args()
    cfg = IniConfigurator(args.config_file, extra_args)

    # Logger setting
    logger = dual_channel_logger(
        __name__,
        file_path=cfg.LOG_FILE,
        file_model='w',
        formatter='%(asctime)s - %(levelname)s - %(message)s',
        time_formatter='%m-%d %H:%M')
    from eval.script_evaluator import ScriptEvaluator

    # DyNet setting
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import dynet_config
    dynet_config.set(mem=cfg.DYNET_MEM, random_seed=cfg.DYNET_SEED)
    dynet_config.set_gpu()
    import dynet as dy
    from models.token_representation import TokenRepresentation
    from antu.nn.dynet.seq2seq_encoders import DeepBiRNNBuilder, orthonormal_VanillaLSTMBuilder
    from models.graph_nn_decoder import GraphNNDecoder
    from models.jackknife_decoder import JackKnifeGraphNNDecoder

    

    # Build the dataset of the training process
    # Build data reader
    data_reader = PTBReader(
        field_list=['word', 'tag', 'head', 'rel'],
        root='0\t**root**\t_\t**rcpos**\t**rpos**\t_\t0\t**rrel**\t_\t_',
        spacer=r'[\t]',)
    # Build vocabulary with pretrained glove
    vocabulary = Vocabulary()
    g_word, _ = glove_reader(cfg.GLOVE)
    pretrained_vocabs = {'glove': g_word}
    vocabulary.extend_from_pretrained_vocab(pretrained_vocabs)
    # Setup datasets
    datasets_settings = {'train': DatasetSetting(cfg.TRAIN, True),
                         'dev': DatasetSetting(cfg.DEV, False),
                         'test': DatasetSetting(cfg.TEST, False), }
    datasets = PTBDataset(vocabulary, datasets_settings, data_reader)
    counters = {'word': Counter(), 'tag': Counter(), 'rel': Counter()}
    datasets.build_dataset(counters, no_pad_namespace={'rel'}, no_unk_namespace={'rel'})

    # Build model
    # Parameter
    pc = dy.ParameterCollection()
    trainer = dy.AdamTrainer(pc, cfg.LR, cfg.ADAM_BETA1, cfg.ADAM_BETA2, cfg.EPS)

    # Token Representation Layer
    token_repre = TokenRepresentation(pc, cfg, datasets.vocabulary, include_pos=True)
    # BiLSTM Encoder Layer
    #encoder = BiaffineAttention()
    #encoder = MultiHeadedAttention(pc, 10, token_repre.token_dim)

    models = [{} for i in range(6)]

    for i in range(3):
        models[i]['token'] = TokenRepresentation(pc, cfg, datasets.vocabulary, include_pos=True)
        models[i]['encoder'] =  Encoder(None, token_repre.token_dim,
                    num_layers=1, num_heads=2, d_kv = 32, d_ff=1024, d_l=112,
                    d_positional=None,
                    num_layers_position_only=0,
                    relu_dropout=0.1, residual_dropout=0.1, attention_dropout=0.1,
                    use_lal=True,
                    lal_d_kv=128,
                    lal_d_proj=128,
                    lal_resdrop=True,
                    lal_pwff=True,
                    lal_q_as_matrix=False,
                    lal_partitioned=True,
                    model=pc)
        models[i]['decoder'] = GraphNNDecoder(pc, cfg, datasets.vocabulary)
        #models[i]['decoder'] = JackKnifeGraphNNDecoder(pc, cfg, datasets.vocabulary)
    
    for i in range(3, 6):
        models[i]['token'] = TokenRepresentation(pc, cfg, datasets.vocabulary, include_pos=True)
        models[i]['encoder'] = DeepBiRNNBuilder(pc, cfg.ENC_LAYERS, token_repre.token_dim,
                                                cfg.ENC_H_DIM, orthonormal_VanillaLSTMBuilder)
        models[i]['decoder'] = GraphNNDecoder(pc, cfg, datasets.vocabulary)
        #models[i]['decoder'] = JackKnifeGraphNNDecoder(pc, cfg, datasets.vocabulary)


    #encoder = MultiLayerMultiHeadAttention(pc, 10, token_repre.token_dim, num_layers=cfg.ENC_LAYERS)
    # encoder = DeepBiRNNBuilder(pc, cfg.ENC_LAYERS, token_repre.token_dim,
    #                            cfg.ENC_H_DIM, orthonormal_VanillaLSTMBuilder)
    # GNN Decoder Layer
    #decoder = GraphNNDecoder(pc, cfg, datasets.vocabulary)
    # PTB Evaluator
    my_eval = ScriptEvaluator(['Valid', 'Test'], datasets.vocabulary)

    # Build Training Batch
    def cmp(ins):
        return len(ins['word'])
    train_batch = datasets.get_batches('train', cfg.TRAIN_BATCH_SIZE, True, cmp, True)
    valid_batch = list(datasets.get_batches('dev', cfg.TEST_BATCH_SIZE, False, cmp, False))
    test_batch = list(datasets.get_batches('test', cfg.TEST_BATCH_SIZE, False, cmp, False))




    # Train model
    BEST_DEV_LAS = BEST_DEV_UAS = BEST_ITER = 0
    cnt_iter = -cfg.WARM * cfg.GRAPH_LAYERS
    valid_loss = [[] for i in range(cfg.GRAPH_LAYERS+3)]
    logger.info("Experiment name: %s" % args.name)
    SHA = os.popen('git log -1 | head -n 1 | cut -c 8-13').readline().rstrip()
    logger.info('Git SHA: %s' % SHA)




    while cnt_iter < cfg.MAX_ITER:
        dy.renew_cg(immediate_compute = True, check_validity = True)
        cnt_iter += 1
        indexes, masks, truth = train_batch.__next__()

        
        for k, model in enumerate(models):
            # only train on 5/6 of the training data
            vectors = model['token'](indexes, True)
            if type(vectors) == list and vectors[0].dim()[1] <= 1:
                continue
            if cnt_iter % 6 == k:
                continue
            
            if k <= 2:
                new_vectors = model['encoder'](vectors, np.array(masks['1D']).T)
            else:
                new_vectors = model['encoder'](vectors, None, cfg.RNN_DROP, cfg.RNN_DROP, np.array(masks['1D']).T, False, True)


        #vectors = token_repre(indexes, True)
        #vectors = encoder(vectors, vectors, vectors, np.array(masks['1D']).T, cfg.RNN_DROP)
        #vectors = encoder(vectors, None, cfg.RNN_DROP, cfg.RNN_DROP, np.array(masks['1D']).T, False, True)
 
            loss, part_loss = model['decoder'](new_vectors, masks, truth, cnt_iter, True, True)
            for i, l in enumerate([loss]+part_loss):
                valid_loss[i].append(l.value())
            loss.backward()
            trainer.learning_rate = cfg.LR*cfg.LR_DECAY**(max(cnt_iter, 0)/cfg.LR_ANNEAL)
            trainer.update()

            if cnt_iter % cfg.VALID_ITER: continue
            # Validation
            for i in range(len(valid_loss)):
                valid_loss[i] = str(round(np.mean(valid_loss[i]), 2))
            avg_loss = ', '.join(valid_loss)
            logger.info("")
            logger.info("Iter: %d-%d, Model index: %d Avg_loss: %s, LR (%f), Best (%d)" %
                        (cnt_iter/cfg.VALID_ITER, cnt_iter, k, avg_loss,
                        trainer.learning_rate, BEST_ITER))

            valid_loss = [[] for i in range(cfg.GRAPH_LAYERS+3)]
            my_eval.clear('Valid')

            
            for indexes, masks, truth in valid_batch:
                dy.renew_cg()

                for k, model in enumerate(models):
                    vectors = model['token'](indexes, False)
                    if type(vectors) == list and vectors[0].dim()[1] <= 1:
                        my_eval.add_truth('Valid', truth)
                        my_eval.add_pred('Valid', truth)
                        continue
                    if k <= 2:                          
                        new_vectors = model['encoder'](vectors, np.array(masks['1D']).T)
                    else:
                        new_vectors = model['encoder'](vectors, None, cfg.RNN_DROP, cfg.RNN_DROP, np.array(masks['1D']).T, False, True)                                  

                    #vectors = token_repre(indexes, False)
                    #vectors = encoder(vectors, vectors, vectors, np.array(masks['1D']).T, cfg.RNN_DROP)
                    #vectors = encoder(vectors, None, cfg.RNN_DROP, cfg.RNN_DROP, np.array(masks['1D']).T, False, False)
                    pred = model['decoder'](new_vectors, masks, None, cnt_iter, False, True)
                    my_eval.add_truth('Valid', truth)
                    my_eval.add_pred('Valid', pred)
            
            dy.save(cfg.LAST_FILE, [model['token'], model['encoder'], model['decoder']])
            if my_eval.evaluation('Valid', cfg.PRED_DEV, cfg.DEV):
                BEST_ITER = cnt_iter/cfg.VALID_ITER
                os.system('cp %s.data %s.data' % (cfg.LAST_FILE, cfg.BEST_FILE))
                os.system('cp %s.meta %s.meta' % (cfg.LAST_FILE, cfg.BEST_FILE))

            # Just record test result
            my_eval.clear('Test')
            for indexes, masks, truth in test_batch:
                dy.renew_cg()
                
                for k, model in enumerate(models):
                    if type(vectors) == list and vectors[0].dim()[1] <= 1:
                        my_eval.add_truth('Test', truth)
                        my_eval.add_pred('Test', truth)
                        continue
                    vectors = model['token'](indexes, False)
                    if k <= 2:
                        new_vectors = model['encoder'](vectors, np.array(masks['1D']).T)
                    else:
                        new_vectors = model['encoder'](vectors, None, cfg.RNN_DROP, cfg.RNN_DROP, np.array(masks['1D']).T, False, True)

                #vectors = token_repre(indexes, False)
                #vectors = encoder(vectors, vectors, vectors, np.array(masks['1D']).T, cfg.RNN_DROP)
                #vectors = encoder(vectors, None, cfg.RNN_DROP, cfg.RNN_DROP, np.array(masks['1D']).T, False, False)
                    pred = model['decoder'](new_vectors, masks, None, cnt_iter, False, True)
                    my_eval.add_truth('Test', truth)
                    my_eval.add_pred('Test', pred)
            my_eval.evaluation('Test', cfg.PRED_TEST, cfg.TEST)
        my_eval.print_best_result('Valid')

    # Final Test
    # TODO: 
    print('done')
    test_pc = dy.ParameterCollection()
    token_repre, encoder, decoder = dy.load(cfg.BEST_FILE, test_pc)
    my_eval.clear('Test')
    for indexes, masks, truth in test_batch:
        dy.renew_cg()
        #vectors = token_repre(indexes, False)
        
        intermediate_vectors_list = []

        for k, model in enumerate(models):
            vectors = model['token'](indexes, False)
            if type(vectors) == list and vectors[0].dim()[1] <= 1:
                continue
            if k <= 2:
                intermediate_vectors = model['encoder'](vectors, np.array(masks['1D']).T)
            else:
                intermediate_vectors = model['encoder'](vectors, None, cfg.RNN_DROP, cfg.RNN_DROP, np.array(masks['1D']).T, False, True)
            
            if type(intermediate_vectors) == list:
                sent_len = len(intermediate_vectors)
                batch_size = intermediate_vectors[0].dim()[1]
                intermediate_vectors = dy.concatenate_cols(intermediate_vectors)
                intermediate_vectors = dy.transpose(intermediate_vectors, [1, 0])
            
            intermediate_vectors_list.append(intermediate_vectors.npvalue())


        if len(intermediate_vectors_list) == 0:
            my_eval.add_truth('Test', truth)
            my_eval.add_pred('Test', truth)
            continue
        average_vector = np.nanmean(intermediate_vectors_list, axis=0)
        vectors = dy.inputTensor(average_vector, batched=True)
        #vectors = encoder(vectors, None, cfg.RNN_DROP, cfg.RNN_DROP, np.array(masks['1D']).T, False, False)
        pred = decoder(vectors, masks, None, 0, False, True)
        my_eval.add_truth('Test', truth)
        my_eval.add_pred('Test', pred)
    my_eval.evaluation('Test', cfg.PRED_TEST, cfg.TEST)


if __name__ == '__main__':
    main()
