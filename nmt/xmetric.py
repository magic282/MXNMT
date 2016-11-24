import numpy as np


# Evaluation
def Perplexity(label, pred):
    label = label.T.reshape((-1,))
    loss = 0.
    mask_count = 0
    for i in range(pred.shape[0]):
        if int(label[i]) == 0:
            mask_count += 1
            continue
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss / (label.size - mask_count))


def MyCrossEntropy(label, pred):
    label = label.T.reshape((-1,))
    loss = 0.
    for i in range(pred.shape[0]):
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return loss / label.size


def get_bleu(gold, test):
    import subprocess
    bleu_computer = r"CompBleu_new.exe"
    rawoutput = subprocess.check_output([bleu_computer, gold, test])
    output = rawoutput.splitlines()
    bleu = float(output[-1].decode('utf-8').split('=')[-1].strip())
    return rawoutput, bleu
