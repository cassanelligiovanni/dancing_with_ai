import os
import torch
import wandb
import json
import numpy as np
from evaluation.accuracy import *
from evaluation.kinetic import *
from interpolate_to_60fps import *

def interpolate60(preds):

    final = preds
    final = final.reshape(final.shape[0], 51)
    root = final[:,3*11:3*12]
    final = final + np.tile(root,(1,17))
    final[:,3*11:3*12] = root
    final = final.reshape(final.shape[0], 17, 3)

    interpolated = np.array(interpolate(final))

    return interpolated.reshape(interpolated.shape[0], 51)



def test_log(model, data_dir, device):

    test_dir = data_dir + "dataset/test"

    names = [fname for fname in os.listdir(test_dir) if fname.endswith('.json')]
    paths = [os.path.join(test_dir, fname) for fname in names]

    model.eval()

    corrects = 0
    beat_coverage = 0

    test_size = len(paths)


    with torch.no_grad():

        for path in paths:

            with open(path) as f :

                file = json.loads(f.read())
                name = file["id"]
                music = np.array(file["music"])

                music_tensor = torch.FloatTensor(music[None, :, :])
                pos =torch.LongTensor(np.arange(1, 143, 1))

                if torch.cuda.is_available() :
                    music_tensor = torch.FloatTensor(music[None, :, :]).to(device)
                    pos =torch.LongTensor(np.arange(1, 143, 1)).to(device)
                    predicted = model.module.predict(music_tensor, pos)
                else :
                    predicted = model.predict(music_tensor, pos)

                loss, correct = get_accuracy(predicted, name)

                corrects += correct

                predicted = np.array([np.array(x.cpu()) for x in predicted])
                interpolated = interpolate60(predicted)

                kinetic_beats = calculate_rom(interpolated[20:, :])
                music_beats = np.nonzero(music[:, 54])

                beat_coverage += len(kinetic_beats)/len(music_beats[0])


    wandb.log({"test ACC(%)": (corrects*100)/test_size, "test Beat Coverage(%)": (beat_coverage*100)/test_size})
