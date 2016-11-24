# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys

sys.path.insert(0, "../../python")
import numpy as np
import mxnet as mx


# The interface of a data iter that works for bucketing
#
# DataIter
#   - default_bucket_key: the bucket key for the default symbol.
#
# DataBatch
#   - provide_data: same as DataIter, but specific to this batch
#   - provide_label: same as DataIter, but specific to this batch
#   - bucket_key: the key for the bucket that should be used for this batch

def default_read_content(path):
    with open(path) as ins:
        content = ins.read()
        content = content.replace('\n', ' <eos> ').replace('. ', ' <eos> ')
        return content


def default_build_vocab(path):
    content = default_read_content(path)
    content = content.split(' ')
    the_vocab = {}
    idx = 1  # 0 is left for zero-padding
    the_vocab[' '] = 0  # put a dummy element here so that len(vocab) is correct
    for word in content:
        if len(word) == 0:
            continue
        if not word in the_vocab:
            the_vocab[word] = idx
            idx += 1
    return the_vocab


def default_text2id(sentence, the_vocab):
    words = sentence.split(' ')
    words = [the_vocab[w] for w in words if len(w) > 0]
    return words


def default_gen_buckets(sentences, batch_size, the_vocab):
    len_dict = {}
    max_len = -1
    for sentence in sentences:
        words = default_text2id(sentence, the_vocab)
        if len(words) == 0:
            continue
        if len(words) > max_len:
            max_len = len(words)
        if len(words) in len_dict:
            len_dict[len(words)] += 1
        else:
            len_dict[len(words)] = 1
    print(len_dict)

    tl = 0
    buckets = []
    for l, n in len_dict.items():  # TODO: There are better heuristic ways to do this
        if n + tl >= batch_size:
            buckets.append(l)
            tl = 0
        else:
            tl += n
    if tl > 0:
        buckets.append(max_len)
    return buckets


class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label, bucket_key):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.bucket_key = bucket_key

        self.pad = 0
        self.index = None  # TODO: what is index?

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


class DummyIter(mx.io.DataIter):
    '''A dummy iterator that always return the same batch, used for speed testing'''

    def __init__(self, real_iter):
        super(DummyIter, self).__init__()
        self.real_iter = real_iter
        self.provide_data = real_iter.provide_data
        self.provide_label = real_iter.provide_label
        self.batch_size = real_iter.batch_size

        for batch in real_iter:
            self.the_batch = batch
            break

    def __iter__(self):
        return self

    def next(self):
        return self.the_batch


class MaskedBucketSentenceIter(mx.io.DataIter):
    def __init__(self, source_path, target_path, source_vocab, target_vocab,
                 buckets, batch_size,
                 source_init_states, target_init_states,
                 source_data_name='source', source_mask_name='source_mask',
                 target_data_name='target', target_mask_name='target_mask',
                 label_name='target_softmax_label',
                 seperate_char=' <eos> ', text2id=None, read_content=None, max_read_sample=sys.maxsize):
        super(MaskedBucketSentenceIter, self).__init__()

        if text2id is None:
            self.text2id = default_text2id
        else:
            self.text2id = text2id
        if read_content is None:
            self.read_content = default_read_content
        else:
            self.read_content = read_content
        source_sentences = self.read_content(source_path, max_read_sample)
        # source_sentences = source_content.split(seperate_char)

        target_sentences = self.read_content(target_path, max_read_sample)
        # target_sentences = target_content.split(seperate_char)

        assert len(source_sentences) == len(target_sentences)

        self.source_vocab_size = len(source_vocab)
        self.target_vocab_size = len(target_vocab)
        self.source_data_name = source_data_name
        self.target_data_name = target_data_name
        self.label_name = label_name

        self.source_mask_name = source_mask_name
        self.target_mask_name = target_mask_name

        buckets.sort()
        self.buckets = buckets
        self.source_data = [[] for _ in buckets]
        self.target_data = [[] for _ in buckets]
        self.label_data = [[] for _ in buckets]
        self.source_mask_data = [[] for _ in buckets]
        self.target_mask_data = [[] for _ in buckets]

        # pre-allocate with the largest bucket for better memory sharing
        self.default_bucket_key = max(buckets)

        num_of_data = len(source_sentences)
        for i in range(num_of_data):
            source = source_sentences[i]
            target = ['<s>'] + target_sentences[i]
            label = target_sentences[i] + ['</s>']
            source_sentence = self.text2id(source, source_vocab)
            target_sentence = self.text2id(target, target_vocab)
            label_id = self.text2id(label, target_vocab)
            if len(source_sentence) == 0 or len(target_sentence) == 0:
                continue
            for j, bkt in enumerate(buckets):
                if bkt[0] >= len(source) and bkt[1] >= len(target):
                    self.source_data[j].append(source_sentence)
                    self.target_data[j].append(target_sentence)
                    self.label_data[j].append(label_id)
                    break
                    # we just ignore the sentence it is longer than the maximum
                    # bucket size here

        # convert data into ndarrays for better speed during training
        source_data = [np.zeros((len(x), buckets[i][0])) for i, x in enumerate(self.source_data)]
        source_mask_data = [np.zeros((len(x), buckets[i][0])) for i, x in enumerate(self.source_data)]
        target_data = [np.zeros((len(x), buckets[i][1])) for i, x in enumerate(self.target_data)]
        target_mask_data = [np.zeros((len(x), buckets[i][1])) for i, x in enumerate(self.target_data)]
        label_data = [np.zeros((len(x), buckets[i][1])) for i, x in enumerate(self.label_data)]
        for i_bucket in range(len(self.buckets)):
            for j in range(len(self.source_data[i_bucket])):
                source = self.source_data[i_bucket][j]
                target = self.target_data[i_bucket][j]
                label = self.label_data[i_bucket][j]
                source_data[i_bucket][j, :len(source)] = source
                source_mask_data[i_bucket][j, :len(source)] = 1
                target_data[i_bucket][j, :len(target)] = target
                target_mask_data[i_bucket][j, :len(target)] = 1
                label_data[i_bucket][j, :len(label)] = label
        self.source_data = source_data
        self.source_mask_data = source_mask_data
        self.target_data = target_data
        self.target_mask_data = target_mask_data
        self.label_data = label_data

        # Get the size of each bucket, so that we could sample
        # uniformly from the bucket
        bucket_sizes = [len(x) for x in self.source_data]

        print("Summary of dataset ==================")
        print('Total: {0} in {1} buckets'.format(num_of_data, len(buckets)))
        for bkt, size in zip(buckets, bucket_sizes):
            print("bucket of {0} : {1} samples".format(bkt, size))

        self.batch_size = batch_size
        self.make_data_iter_plan()

        self.source_init_states = source_init_states
        self.target_init_states = target_init_states
        self.source_init_state_arrays = [mx.nd.zeros(x[1]) for x in source_init_states]
        self.target_init_state_arrays = [mx.nd.zeros(x[1]) for x in target_init_states]

        self.provide_data = [(source_data_name, (batch_size, self.default_bucket_key[0])),
                             (source_mask_name, (batch_size, self.default_bucket_key[0])),
                             (target_data_name, (batch_size, self.default_bucket_key[1])),
                             (target_mask_name, (batch_size, self.default_bucket_key[1]))] \
                            + source_init_states + target_init_states
        self.provide_label = [(label_name, (self.batch_size, self.default_bucket_key[1]))]

    def make_data_iter_plan(self):
        "make a random data iteration plan"
        # truncate each bucket into multiple of batch-size
        bucket_n_batches = []
        for i in range(len(self.source_data)):
            bucket_n_batches.append(len(self.source_data[i]) / self.batch_size)
            self.source_data[i] = self.source_data[i][:int(bucket_n_batches[i] * self.batch_size)]
            self.source_mask_data[i] = self.source_mask_data[i][:int(bucket_n_batches[i] * self.batch_size)]
            self.target_data[i] = self.target_data[i][:int(bucket_n_batches[i] * self.batch_size)]
            self.target_mask_data[i] = self.target_mask_data[i][:int(bucket_n_batches[i] * self.batch_size)]

        bucket_plan = np.hstack([np.zeros(n, int) + i for i, n in enumerate(bucket_n_batches)])
        np.random.shuffle(bucket_plan)

        bucket_idx_all = [np.random.permutation(len(x)) for x in self.source_data]

        self.bucket_plan = bucket_plan
        self.bucket_idx_all = bucket_idx_all
        self.bucket_curr_idx = [0 for x in self.source_data]

        self.source_data_buffer = []
        self.source_mask_data_buffer = []
        self.target_data_buffer = []
        self.target_mask_data_buffer = []
        self.label_buffer = []
        for i_bucket in range(len(self.source_data)):
            source_data = np.zeros((self.batch_size, self.buckets[i_bucket][0]))
            source_mask_data = np.zeros((self.batch_size, self.buckets[i_bucket][0]))
            target_data = np.zeros((self.batch_size, self.buckets[i_bucket][1]))
            target_mask_data = np.zeros((self.batch_size, self.buckets[i_bucket][1]))
            label = np.zeros((self.batch_size, self.buckets[i_bucket][1]))

            self.source_data_buffer.append(source_data)
            self.source_mask_data_buffer.append(source_mask_data)
            self.target_data_buffer.append(target_data)
            self.target_mask_data_buffer.append(target_mask_data)
            self.label_buffer.append(label)

    def __iter__(self):
        source_init_state_names = [x[0] for x in self.source_init_states]
        target_init_state_names = [x[0] for x in self.target_init_states]

        for i_bucket in self.bucket_plan:
            source_data = self.source_data_buffer[i_bucket]
            source_mask_data = self.source_mask_data_buffer[i_bucket]
            target_data = self.target_data_buffer[i_bucket]
            target_mask_data = self.target_mask_data_buffer[i_bucket]
            label = self.label_buffer[i_bucket]

            i_idx = self.bucket_curr_idx[i_bucket]
            idx = self.bucket_idx_all[i_bucket][i_idx:i_idx + self.batch_size]
            self.bucket_curr_idx[i_bucket] += self.batch_size
            source_data[:] = self.source_data[i_bucket][idx]
            source_mask_data[:] = self.source_mask_data[i_bucket][idx]
            target_data[:] = self.target_data[i_bucket][idx]
            target_mask_data[:] = self.target_mask_data[i_bucket][idx]
            label[:] = self.label_data[i_bucket][idx]

            data_all = [mx.nd.array(source_data), mx.nd.array(source_mask_data)] + \
                       [mx.nd.array(target_data), mx.nd.array(target_mask_data)] + \
                       self.source_init_state_arrays + self.target_init_state_arrays
            label_all = [mx.nd.array(label)]
            data_names = [self.source_data_name, self.source_mask_name] + [
                self.target_data_name, self.target_mask_name] + source_init_state_names + target_init_state_names
            label_names = [self.label_name]

            data_batch = SimpleBatch(data_names, data_all, label_names, label_all,
                                     self.buckets[i_bucket])
            yield data_batch

    def reset(self):
        self.bucket_curr_idx = [0 for x in self.source_data]
