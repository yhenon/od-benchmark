import numpy as np


def postprocess(outputs):
    return int(np.argmax(outputs["logits"], axis=1)[0])
