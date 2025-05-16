import json
import matplotlib.pyplot as plt
import os

JSON_PATH = './checkpoints/train_json/'
EVAL_PATH = './checkpoints/eval_json/'
MODE = ['e2e', 'frozen']
DATA_SIZE = ['2000', '160', '80', '40', '20', '10']
SAVE_PATH = './checkpoints/train_images/'

train_info = dict()
eval_info = dict()

for m in MODE:
    for ds in DATA_SIZE:
        json_name = f'seg_{m}_{ds}.json'
        json_path = os.path.join(JSON_PATH, json_name)
        with open(json_path, 'r') as f:
            train_info[(m, ds)] = json.load(f)
        json_path = os.path.join(EVAL_PATH, json_name)
        with open(json_path, 'r') as f:
            eval_info[(m, ds)] = json.load(f)

for ds in DATA_SIZE:
    train_loss_e2e = train_info[('e2e', ds)]['train_loss']
    val_loss_e2e = train_info[('e2e', ds)]['val_loss']
    train_loss_frozen = train_info[('frozen', ds)]['train_loss']
    val_loss_frozen = train_info[('frozen', ds)]['val_loss']

    epoch = train_info[('e2e', ds)]['epoch']
    best_epoch_e = eval_info[('e2e', ds)]['epoch'] - 1
    best_epoch_f = eval_info[('frozen', ds)]['epoch'] - 1

    plt.xticks(ticks=range(0, epoch, epoch // 10))

    plt.plot(train_loss_e2e, label='train loss E'+ds, color='orange', linestyle='-.', linewidth=0.7)
    plt.plot(train_loss_frozen, label='train loss FP'+ds, color='blue', linestyle='-.', linewidth=0.7)
    plt.plot(val_loss_e2e, label='val loss E'+ds, color='orange', linestyle='-', linewidth=0.7)
    plt.plot(val_loss_frozen, label='val loss FP'+ds, color='blue', linestyle='-', linewidth=0.7)

    plt.axvline(x=best_epoch_e, color='orange', linestyle='--', linewidth=1)
    plt.axvline(x=best_epoch_f, color='blue', linestyle='--', linewidth=1)
    # plt.text(best_epoch_e, plt.ylim()[1] * 0.9, 'Split Point', ha='center', va='bottom', fontsize=9)
    # plt.text(best_epoch_f, plt.ylim()[1] * 0.9, 'Split Point', ha='center', va='bottom', fontsize=9)


    plt.legend()
    plt.xlabel("Epochs")
    plt.grid(True)
    plt.savefig(SAVE_PATH+"Loss_curves_"+ds+".png", dpi=300, bbox_inches='tight')
    plt.clf()