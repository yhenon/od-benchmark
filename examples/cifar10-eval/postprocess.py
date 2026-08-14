import numpy as np


def postprocess(outputs):
    logits = outputs["logits"]
    return int(np.argmax(logits, axis=1)[0])
