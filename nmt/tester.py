import xconfig
from inference import BiS2SInferenceModel
from inference_mask import BiS2SInferenceModel_mask
from xutils import read_content, load_vocab, sentence2id, word2id

import mxnet as mx
import numpy as np
import logging
import random
import bisect
from collections import OrderedDict, namedtuple
from mxwrap.rnn.LSTM import LSTMState

BeamNode = namedtuple("BeamNode", ["father", "content", "score", "acc_score", "finish", "finishLen"])

random_sample = False


def get_inference_models(buckets, arg_params, source_vocab_size, target_vocab_size, ctx, batch_size):
    # build an inference model
    model_buckets = OrderedDict()
    for bucket in buckets:
        model_buckets[bucket] = BiS2SInferenceModel_mask(s_num_lstm_layer=xconfig.num_lstm_layer, s_seq_len=bucket[0],
                                                         s_vocab_size=source_vocab_size + 1,
                                                         s_num_hidden=xconfig.num_hidden, s_num_embed=xconfig.num_embed,
                                                         s_dropout=0,
                                                         t_num_lstm_layer=xconfig.num_lstm_layer, t_seq_len=bucket[1],
                                                         t_vocab_size=target_vocab_size + 1,
                                                         t_num_hidden=xconfig.num_hidden, t_num_embed=xconfig.num_embed,
                                                         t_num_label=target_vocab_size + 1, t_dropout=0,
                                                         arg_params=arg_params,
                                                         use_masking=True,
                                                         ctx=ctx, batch_size=batch_size)
    return model_buckets


def get_bucket_model(model_buckets, input_len):
    for bucket, m in model_buckets.items():
        if bucket[0] >= input_len:
            return m
    return None


# helper strcuture for prediction
def MakeRevertVocab(vocab):
    dic = {}
    for k, v in vocab.items():
        dic[v] = k
    return dic


# make input from char
def MakeInput(sentence, vocab, unroll_len, data_arr, mask_arr):
    idx = sentence2id(sentence, vocab)
    tmp = np.zeros((1, unroll_len))
    mask = np.zeros((1, unroll_len))
    for i in range(min(len(idx), unroll_len)):
        tmp[0][i] = idx[i]
        mask[0][i] = 1
    data_arr[:] = tmp
    mask_arr[:] = mask


def MakeInput_beam(sentence, vocab, unroll_len, data_arr, mask_arr, beam_size):
    idx = sentence2id(sentence, vocab)
    tmp = np.zeros((beam_size, unroll_len))
    mask = np.zeros((beam_size, unroll_len))
    for i in range(min(len(idx), unroll_len)):
        for j in range(beam_size):
            tmp[j][i] = idx[i]
            mask[j][i] = 1
    data_arr[:] = tmp
    mask_arr[:] = mask


def MakeInput_batch(sentences, vocab, unroll_len, data_arr, mask_arr, batch_size):
    tmp = np.zeros((batch_size, unroll_len))
    mask = np.zeros((batch_size, unroll_len))
    actual_sample_num = len(sentences)
    for i in range(min(batch_size, actual_sample_num)):
        idx = sentence2id(sentences[i], vocab)
        for j in range(min(len(idx), unroll_len)):
            tmp[i][j] = idx[j]
            mask[i][j] = 1
    data_arr[:] = tmp
    mask_arr[:] = mask


def MakeTargetInput(char, vocab, arr):
    idx = word2id(char, vocab)
    tmp = np.zeros((1,))
    tmp[0] = idx
    arr[:] = tmp


def MakeTargetInput_batch(chars, vocab, arr, batch_size):
    tmp = np.zeros((batch_size,))
    actual_sample_num = len(chars)
    for idx in range(min(batch_size, actual_sample_num)):
        word_id = word2id(chars[idx], vocab)
        tmp[idx] = word_id
    arr[:] = tmp


def MakeTargetInput_beam(beam_nodes, vocab, arr):
    tmp = np.zeros((len(beam_nodes),))
    for idx in range(len(beam_nodes)):
        word_id = vocab[beam_nodes[idx].content] if beam_nodes[idx].content in vocab else vocab['<unk>']
        tmp[idx] = word_id
    arr[:] = tmp


# helper function for random sample
def _cdf(weights):
    total = sum(weights)
    result = []
    cumsum = 0
    for w in weights:
        cumsum += w
        result.append(cumsum / total)
    return result


def _choice(population, weights):
    assert len(population) == len(weights)
    cdf_vals = _cdf(weights)
    x = random.random()
    idx = bisect.bisect(cdf_vals, x)
    return population[idx]


# we can use random output or fixed output by choosing largest probability
def MakeOutput(prob, vocab, sample=False, temperature=1.):
    if sample == False:
        idx = np.argmax(prob, axis=1)[0]
    else:
        fix_dict = [""] + [vocab[i] for i in range(1, len(vocab) + 1)]
        scale_prob = np.clip(prob, 1e-6, 1 - 1e-6)
        rescale = np.exp(np.log(scale_prob) / temperature)
        rescale[:] /= rescale.sum()
        return _choice(fix_dict, rescale[0, :])
    try:
        char = vocab[idx]
    except:
        char = ''
    return char


# we can use random output or fixed output by choosing largest probability
def MakeOutput_batch(probs, vocab, sample=False, temperature=1.):
    res = []
    for i in range(probs.shape[0]):
        prob = probs[i]
        if sample == False:
            idx = np.argmax(prob)
        else:
            fix_dict = [""] + [vocab[i] for i in range(1, len(vocab) + 1)]
            scale_prob = np.clip(prob, 1e-6, 1 - 1e-6)
            rescale = np.exp(np.log(scale_prob) / temperature)
            rescale[:] /= rescale.sum()
            return _choice(fix_dict, rescale[0, :])
        try:
            char = vocab[idx]
        except:
            char = ''
        res.append(char)
    return res


def translate_one(max_decode_len, sentence, model_buckets, unroll_len, source_vocab, target_vocab, revert_vocab,
                  target_ndarray):
    input_length = len(sentence)
    cur_model = get_bucket_model(model_buckets, input_length)
    input_ndarray = mx.nd.zeros((1, unroll_len))
    mask_ndarray = mx.nd.zeros((1, unroll_len))
    output = ['<s>']
    MakeInput(sentence, source_vocab, unroll_len, input_ndarray, mask_ndarray)
    last_encoded, all_encoded = cur_model.encode(input_ndarray,
                                                 mask_ndarray)  # last_encoded means the last time step hidden
    for i in range(max_decode_len):
        MakeTargetInput(output[-1], target_vocab, target_ndarray)
        prob, attention_weights = cur_model.decode_forward(last_encoded, all_encoded, mask_ndarray, target_ndarray,
                                                           i == 0)
        next_char = MakeOutput(prob, revert_vocab, random_sample)
        if next_char == '</s>':
            break
        output.append(next_char)
    return output[1:]


def translate_greedy_batch(max_decode_len, sentences, batch_size, model_buckets, unroll_len, source_vocab, target_vocab,
                           revert_vocab, target_ndarray):
    cur_model = get_bucket_model(model_buckets, unroll_len)
    input_ndarray = mx.nd.zeros((batch_size, unroll_len))
    mask_ndarray = mx.nd.zeros((batch_size, unroll_len))
    output = [['<s>'] * batch_size]
    MakeInput_batch(sentences, source_vocab, unroll_len, input_ndarray, mask_ndarray, batch_size)
    last_encoded, all_encoded = cur_model.encode(input_ndarray,
                                                 mask_ndarray)  # last_encoded means the last time step hidden
    for i in range(max_decode_len):
        MakeTargetInput_batch(output[-1], target_vocab, target_ndarray, batch_size)
        probs, attention_weights = cur_model.decode_forward(last_encoded, all_encoded, mask_ndarray, target_ndarray,
                                                            i == 0)
        next_chars = MakeOutput_batch(probs, revert_vocab, random_sample)
        finished = [ch == '</s>' for ch in next_chars]
        if all(finished):
            break
        output.append(next_chars)
    return output[1:]


def _smallest(matrix, k, only_first_row=False):
    """Find k smallest elements of a matrix.

    Parameters
    ----------
    matrix : :class:`numpy.ndarray`
        The matrix.
    k : int
        The number of smallest elements required.
    only_first_row : bool, optional
        Consider only elements of the first row.

    Returns
    -------
    Tuple of ((row numbers, column numbers), values).

    """
    if only_first_row:
        flatten = matrix[:1, :].flatten()
    else:
        flatten = matrix.flatten()
    # flatten = -flatten
    args = np.argpartition(flatten, k)[:k]
    args = args[np.argsort(flatten[args])]
    return np.unravel_index(args, matrix.shape), flatten[args]


def translate_one_with_beam(max_decode_len, sentence, model_buckets, unroll_len, source_vocab, target_vocab,
                            revert_vocab, target_ndarray, beam_size, eos_index):
    input_length = len(sentence)
    cur_model = get_bucket_model(model_buckets, input_length)
    input_ndarray = mx.nd.zeros((beam_size, unroll_len))
    mask_ndarray = mx.nd.zeros((beam_size, unroll_len))

    beam = [[BeamNode(father=-1, content='<s>', score=0.0, acc_score=0.0, finish=False, finishLen=0) for i in
             range(beam_size)]]
    beam_state = [None]

    MakeInput_beam(sentence, source_vocab, unroll_len, input_ndarray, mask_ndarray, beam_size)
    last_encoded, all_encoded = cur_model.encode(input_ndarray,
                                                 mask_ndarray)  # last_encoded means the last time step hidden
    for i in range(max_decode_len):
        MakeTargetInput_beam(beam[-1], target_vocab, target_ndarray)
        prob, attention_weights, new_state = cur_model.decode_forward_with_state(last_encoded, all_encoded,
                                                                                 mask_ndarray, target_ndarray,
                                                                                 beam_state[-1], i == 0)
        log_prob = -mx.ndarray.log(prob)
        finished_beam = [t for t, x in enumerate(beam[-1]) if x.finish]
        for idx in range(beam_size):
            # log_prob[idx] = mx.nd.add(log_prob[idx], beam[-1][idx].score)
            if not beam[-1][idx].finish:
                # log_prob[idx] += beam[-1][idx].acc_score
                log_prob[idx] = (log_prob[idx] + beam[-1][idx].acc_score * beam[-1][idx].finishLen) / (
                    beam[-1][idx].finishLen + 1)
            else:
                # log_prob[idx] = beam[-1][idx].acc_score
                log_prob[idx] = beam[-1][idx].acc_score
        for idx in finished_beam:
            log_prob[idx][:eos_index] = np.inf
            log_prob[idx][eos_index + 1:] = np.inf

        (indexes, outputs), chosen_costs = _smallest(log_prob.asnumpy(), beam_size, only_first_row=(i == 0))
        next_chars = [revert_vocab[idx] if idx in revert_vocab else '' for idx in outputs]

        next_state_h = mx.nd.empty(new_state.h.shape, ctx=mx.gpu(0))
        next_state_c = mx.nd.empty(new_state.c.shape, ctx=mx.gpu(0))
        for idx in range(beam_size):
            next_state_h[idx] = new_state.h[np.asscalar(indexes[idx])]
            next_state_c[idx] = new_state.c[np.asscalar(indexes[idx])]
        next_state = LSTMState(c=next_state_c, h=next_state_h)
        beam_state.append(next_state)

        next_beam = [BeamNode(father=indexes[idx],
                              content=next_chars[idx] if not beam[-1][indexes[idx]].finish else beam[-1][
                                  indexes[idx]].content,
                              score=chosen_costs[idx] - beam[-1][indexes[idx]].acc_score,
                              acc_score=chosen_costs[idx],
                              finish=(next_chars[idx] == '</s>' or beam[-1][indexes[idx]].finish),
                              finishLen=(beam[-1][indexes[idx]].finishLen if beam[-1][indexes[idx]].finish else (
                                  beam[-1][indexes[idx]].finishLen + 1))) for
                     idx in range(beam_size)]
        beam.append(next_beam)
        finished = [node.finish for node in beam[-1]]
        if all(finished):
            break
            # output.append(next_char)
    all_result = []
    all_score = []
    for aaa in range(beam_size):
        ptr = aaa
        result = []

        for idx in range(len(beam) - 1 - 1, 0, -1):
            word = beam[idx][ptr].content
            if word != '</s>':
                result.append(word)
            ptr = beam[idx][ptr].father
        result = result[::-1]
        all_result.append(' '.join(result))
        all_score.append(beam[-1][aaa].acc_score)

    return all_result, all_score


def test_on_file_iwslt(input_file, output_file, model_buckets, source_vocab, target_vocab, revert_vocab, ctx,
                       unroll_len,
                       max_decode_len,
                       do_beam=False,
                       beam_size=1):
    beam_file = open(output_file + '_beam', 'w', encoding='utf-8') if do_beam else None
    batch_size = beam_size if do_beam else 1
    eos_index = target_vocab[xconfig.eos_word]
    target_ndarray = mx.nd.zeros((batch_size,), ctx=ctx)
    read_count = 0
    with open(input_file, mode='r', encoding='utf-8') as f, open(output_file, 'w', encoding='utf-8') as of:
        for line in f:
            read_count += 1
            if (read_count - 1) % (xconfig.bleu_ref_number + 2) != 0:
                continue

            ch = line.split(' |||| ')[0].strip().split(' ')
            if do_beam:
                # en = translate_one_with_beam(ch, model_buckets, beam_size)
                all_en, all_score = translate_one_with_beam(max_decode_len, ch, model_buckets, unroll_len, source_vocab,
                                                            target_vocab, revert_vocab, target_ndarray,
                                                            beam_size, eos_index)
                en = all_en[0]
            else:
                en = translate_one(max_decode_len, ch, model_buckets, unroll_len, source_vocab, target_vocab,
                                   revert_vocab,
                                   target_ndarray)
                en = ' '.join(en)
            of.write(en + '\n')
            if do_beam:
                for idx in range(len(all_en)):
                    beam_file.write('{0}\t{1}\n'.format(all_en[idx], all_score[idx]))
                beam_file.write('\n')
    if beam_file:
        beam_file.close()


def test_on_file_greedy_batch_iwslt(input_file, output_file, model_buckets, source_vocab, target_vocab, revert_vocab,
                                    ctx,
                                    unroll_len, max_decode_len, batch_size):
    with open(input_file, mode='r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    lines = lines[0::(xconfig.bleu_ref_number + 2)]
    input_sents = [line.split(' |||| ')[0].strip().split(' ') for line in lines]
    batch_sents = [input_sents[i: i + batch_size] for i in range(0, len(input_sents), batch_size)]
    eos_index = target_vocab[xconfig.eos_word]
    with open(output_file, 'w', encoding='utf-8') as of:
        for batch in batch_sents:
            target_ndarray = mx.nd.zeros((batch_size,), ctx=ctx)
            output_sents = translate_greedy_batch(max_decode_len, batch, batch_size,
                                                  model_buckets, unroll_len, source_vocab,
                                                  target_vocab, revert_vocab, target_ndarray)
            for i in range(len(batch)):
                tmp = []
                for j in range(len(output_sents)):
                    word = output_sents[j][i]
                    if word == xconfig.eos_word:
                        break
                    tmp.append(word)
                of.write(' '.join(tmp) + '\n')


def test():
    # load vocabulary
    source_vocab = load_vocab(xconfig.source_vocab_path, xconfig.special_words)
    target_vocab = load_vocab(xconfig.target_vocab_path, xconfig.special_words)

    revert_vocab = MakeRevertVocab(target_vocab)

    print('source_vocab size: {0}'.format(len(source_vocab)))
    print('target_vocab size: {0}'.format(len(target_vocab)))

    # load model from check-point
    _, arg_params, __ = mx.model.load_checkpoint(xconfig.model_to_load_prefix, xconfig.model_to_load_number)

    buckets = xconfig.buckets
    buckets = [max(buckets)]

    if xconfig.use_batch_greedy_search:
        if xconfig.use_beam_search:
            logging.warning(
                'use_batch_greedy_search and use_beam_search both True, fallback to use_batch_greedy_search')

        model_buckets = get_inference_models(buckets, arg_params, len(source_vocab), len(target_vocab),
                                             xconfig.test_device, batch_size=xconfig.greedy_batch_size)
        test_on_file_greedy_batch_iwslt(input_file=xconfig.test_source, output_file=xconfig.test_output,
                                        model_buckets=model_buckets,
                                        source_vocab=source_vocab, target_vocab=target_vocab, revert_vocab=revert_vocab,
                                        ctx=xconfig.test_device, unroll_len=max(buckets)[0],
                                        max_decode_len=xconfig.max_decode_len, batch_size=xconfig.greedy_batch_size)
    else:
        model_buckets = get_inference_models(buckets, arg_params, len(source_vocab), len(target_vocab),
                                             xconfig.test_device, batch_size=xconfig.beam_size)
        test_on_file_iwslt(input_file=xconfig.test_source, output_file=xconfig.test_output, model_buckets=model_buckets,
                           source_vocab=source_vocab, target_vocab=target_vocab, revert_vocab=revert_vocab,
                           ctx=xconfig.test_device, unroll_len=max(buckets)[0], max_decode_len=xconfig.max_decode_len,
                           do_beam=xconfig.use_beam_search, beam_size=xconfig.beam_size)

    del model_buckets
    from xmetric import get_bleu
    raw_output, scores = get_bleu(xconfig.test_gold, xconfig.test_output)
    logging.info(raw_output)
    logging.info(str(scores))


def test_use_model_param(arg_params, test_file, output_file, gold_file, use_beam=False, beam_size=-1):
    # load vocabulary
    source_vocab = load_vocab(xconfig.source_vocab_path, xconfig.special_words)
    target_vocab = load_vocab(xconfig.target_vocab_path, xconfig.special_words)

    revert_vocab = MakeRevertVocab(target_vocab)

    buckets = xconfig.buckets
    buckets = [max(buckets)]
    b_size = beam_size if use_beam else xconfig.greedy_batch_size
    model_buckets = get_inference_models(buckets, arg_params, len(source_vocab), len(target_vocab),
                                         xconfig.test_device, batch_size=b_size)
    if use_beam:
        test_on_file_iwslt(input_file=test_file, output_file=output_file, model_buckets=model_buckets,
                           source_vocab=source_vocab, target_vocab=target_vocab, revert_vocab=revert_vocab,
                           ctx=xconfig.test_device, unroll_len=max(buckets)[0], max_decode_len=xconfig.max_decode_len,
                           do_beam=use_beam, beam_size=beam_size)
    else:
        test_on_file_greedy_batch_iwslt(input_file=test_file, output_file=output_file, model_buckets=model_buckets,
                                        source_vocab=source_vocab, target_vocab=target_vocab, revert_vocab=revert_vocab,
                                        ctx=xconfig.test_device, unroll_len=max(buckets)[0],
                                        max_decode_len=xconfig.max_decode_len, batch_size=xconfig.greedy_batch_size)
    from xmetric import get_bleu
    raw_output, score = get_bleu(gold_file, output_file)
    logging.info(raw_output)
    del model_buckets
    return score
