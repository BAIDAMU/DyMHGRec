import csv
import math

import dill
import numpy as np
import argparse
from collections import defaultdict
from torch.optim import Adam
import os
import torch
import time
from DyMHGRec_models import HypDrug
from seed import set_seed
from util import llprint, multi_label_metric, ddi_rate_score
import torch.nn.functional as F

# setting
model_name = "Hyp_monmery_dim128"
resume_path = "saved/Hyp_monmery_dim128/Epoch_63_TARGET_0.06_JA_0.6815_DDI_0.04874.model"
# resume_path = "saved/DyMHGRec_01/Epoch_72_TARGET_0.06_JA_0.679_DDI_0.04667.model"
if not os.path.exists(os.path.join("saved", model_name)):
    os.makedirs(os.path.join("saved", model_name))

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument("--Test", action="store_true", default=True, help="test mode")
parser.add_argument("--model_name", type=str, default=model_name, help="model name")
parser.add_argument("--resume_path", type=str, default=resume_path, help="resume path")
parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
parser.add_argument("--target_ddi", type=float, default=0.06, help="target ddi")
parser.add_argument("--dim", type=int, default=128, help="dimension")
parser.add_argument("--cuda", type=int, default=0, help="which cuda")
parser.add_argument("--a", type=float, default=0.9, help="coefficient of a")

args = parser.parse_args()

# evaluate
def eval(model, data_eval, divided_val, voc_size, epoch):
    model.eval()

    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    med_cnt, visit_cnt = 0, 0

    for step, (input, divided_seq) in enumerate(zip(data_eval, divided_val)):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        for adm_idx, (adm, divided_amd) in enumerate(zip(input, divided_seq)):
            target_output, _= model(input[: adm_idx + 1], divided_seq[: adm_idx + 1])

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            # prediction prod
            target_output = F.sigmoid(target_output).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output)

            # prediction med set
            y_pred_tmp = target_output.copy()
            y_pred_tmp[y_pred_tmp >= 0.5] = 1
            y_pred_tmp[y_pred_tmp < 0.5] = 0
            y_pred.append(y_pred_tmp)

            # prediction label
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)

        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
        )

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint("\rtest step: {} / {}".format(step, len(data_eval)))

    # ddi rate
    ddi_rate = ddi_rate_score(smm_record, path="../data/data_new/ddi_A_final.pkl")

    llprint(
        "\nDDI Rate: {:.4}, Jaccard: {:.4},  PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n".format(
            ddi_rate,
            np.mean(ja),
            np.mean(prauc),
            np.mean(avg_p),
            np.mean(avg_r),
            np.mean(avg_f1),
            med_cnt / visit_cnt,
        )
    )

    return (
        ddi_rate,
        np.mean(ja),
        np.mean(prauc),
        np.mean(avg_p),
        np.mean(avg_r),
        np.mean(avg_f1),
        med_cnt / visit_cnt,
    )


def main():

    # load data
    data_path = "../data/data_new/records_final_131.pkl"
    voc_path = "../data/data_new/voc_final.pkl"
    ddi_adj_path = "../data/data_new/ddi_A_final.pkl"
    ddi_mask_path = "../data/data_new/ddi_mask_H.pkl"
    ddi_A_final_path = "../data/data_new/ddi_A_final.pkl"
    ehr_adj_path = '../data/data_new/ehr_adj_final.pkl'

    #数据分类
    all_divided_path = "../data/data_divided/all_divided.pkl"

    device = torch.device("cuda:{}".format(args.cuda))

    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    ddi_adj = dill.load(open(ddi_adj_path, "rb"))
    ddi_mask_H = dill.load(open(ddi_mask_path, "rb"))
    data = dill.load(open(data_path, "rb"))
    
    #读取数据
    patient_divided = dill.load(open(all_divided_path, "rb"))

    voc = dill.load(open(voc_path, "rb"))
    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]
    print(f"Diag num:{len(diag_voc.idx2word)}")
    print(f"Proc num:{len(pro_voc.idx2word)}")
    print(f"Med num:{len(med_voc.idx2word)}")

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point : split_point + eval_len]
    data_eval = data[split_point + eval_len :]
    
    split_point_divided = int(len(patient_divided) * 2 / 3)
    train_divided = patient_divided[:split_point_divided]
    eval_len = int(len(patient_divided[split_point_divided:]) / 2)
    test_divided = patient_divided[split_point_divided : split_point_divided + eval_len]
    eval_divided = patient_divided[split_point_divided + eval_len :]
    

    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    model = HypDrug(
        voc_size,
        ehr_adj,
        ddi_adj,
        ddi_mask_H,
        emb_dim=args.dim,
        device=device,
    )

    if args.Test:
        model.load_state_dict(torch.load(open(args.resume_path, "rb")))
        model.to(device=device)
        torch.manual_seed(97)
        np.random.seed(97)
        tic = time.time()
        sample_size = round(len(data_test) * 0.8)
        result = []
        combined_data = list(zip(data_test, test_divided))
        for _ in range(10):
            # test_sample = np.random.choice(
            #     np.array(data_test, dtype=object), sample_size, replace=True
            # )
            # divided_sample = np.random.choice(
            #     np.array(test_divided, dtype=object), sample_size, replace=True
            # )
            indices = np.random.choice(len(combined_data), sample_size, replace=True)
            # 根据抽取的索引，生成 test_sample 和 divided_sample
            test_sample = [combined_data[i][0] for i in indices]
            divided_sample = [combined_data[i][1] for i in indices]
            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(
                model, test_sample, divided_sample, voc_size, 0
            )
            result.append([ddi_rate, ja, avg_f1, prauc, avg_med])

        result = np.array(result)
        mean = result.mean(axis=0)
        std = result.std(axis=0)

        outstring = ""
        for m, s in zip(mean, std):
            outstring += "{:.4f} $\pm$ {:.4f} & ".format(m, s)

        print(outstring)

        print("test time: {}".format(time.time() - tic))
        return

    model.to(device=device)
    optimizer = Adam(list(model.parameters()), lr=args.lr)

    # start iterations
    history = defaultdict(list)
    best_epoch, best_ja = 0, 0


    EPOCH = 80
    for epoch in range(EPOCH):
        tic = time.time()
        print("\nepoch {} --------------------------".format(epoch + 1))

        model.train()
        for step, (input, divided_seq) in enumerate(zip(data_train, train_divided)):
            loss = 0
            for idx, (adm, divided_adm) in enumerate(zip(input, divided_seq)):

                seq_input = input[: idx + 1]
                seq_divided = divided_seq[: idx + 1]
                
                loss_bce_target = np.zeros((1, voc_size[2]))
                loss_bce_target[:, adm[2]] = 1

                loss_multi_target = np.full((1, voc_size[2]), -1)
                for idx, item in enumerate(adm[2]):
                    loss_multi_target[0][idx] = item

                result, loss_ddi= model(seq_input, seq_divided)      # 新增diag, pro, med,返回sim

                loss_bce = F.binary_cross_entropy_with_logits(
                    result, torch.FloatTensor(loss_bce_target).to(device))

                loss_multi = F.multilabel_margin_loss(
                    F.sigmoid(result), torch.LongTensor(loss_multi_target).to(device)
                )

                result = F.sigmoid(result).detach().cpu().numpy()[0]
                result[result >= 0.5] = 1
                result[result < 0.5] = 0
                y_label = np.where(result == 1)[0]
                current_ddi_rate = ddi_rate_score(
                    [[y_label]], path=ddi_A_final_path
                )

                if current_ddi_rate <= args.target_ddi:
                    loss = 0.95 * loss_bce + 0.05 * loss_multi

                else:
                    beta = max(0, 1 - np.tanh(current_ddi_rate))
                    loss = beta * (0.95 * loss_bce + 0.05 * loss_multi)+ (1 - beta) * loss_ddi 
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            llprint("\rtraining step: {} / {}".format(step, len(data_train)))

        print()
        tic2 = time.time()
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(
            model, data_eval, eval_divided, voc_size, epoch
        )
        print(
            "training time: {}, test time: {}".format(
                time.time() - tic, time.time() - tic2
            )
        )

        history["ja"].append(ja)
        history["ddi_rate"].append(ddi_rate)
        history["avg_p"].append(avg_p)
        history["avg_r"].append(avg_r)
        history["avg_f1"].append(avg_f1)
        history["prauc"].append(prauc)
        history["med"].append(avg_med)

        if epoch >= 5:
            print(
                "ddi: {}, Med: {}, Ja: {}, F1: {}, PRAUC: {}".format(
                    np.mean(history["ddi_rate"][-5:]),
                    np.mean(history["med"][-5:]),
                    np.mean(history["ja"][-5:]),
                    np.mean(history["avg_f1"][-5:]),
                    np.mean(history["prauc"][-5:]),
                )
            )

        torch.save(
            model.state_dict(),
            open(
                os.path.join(
                    "saved",
                    args.model_name,
                    "Epoch_{}_TARGET_{:.2}_JA_{:.4}_DDI_{:.4}.model".format(
                        epoch, args.target_ddi, ja, ddi_rate
                    ),
                ),
                "wb",
            ),
        )

        # if epoch != 0 and best_ja < ja:
        #     best_epoch = epoch+1
        #     best_ja = ja
        if best_ja < ja:
            best_epoch = epoch+1
            best_ddi_rate = round(ddi_rate, 4)
            best_ja = round(ja, 4)
            best_prauc = round(prauc, 4)
            best_avg_p = round(avg_p, 4)
            best_avg_r = round(avg_r, 4)
            best_avg_f1 = round(avg_f1, 4)
            best_avg_med= round(avg_med, 4)

        # print("best_epoch: {}".format(best_epoch))
        print(
            "best_epoch: {}, best_ddi_rate: {}, best_ja: {}, best_avg_f1: {}, best_prauc: {}, best_avg_med: {}, best_avg_p: {}, best_avg_r: {}"
                .format(best_epoch, best_ddi_rate, best_ja, best_avg_f1, best_prauc, best_avg_med, best_avg_p,
                        best_avg_r))

    dill.dump(
        history,
        open(
            os.path.join(
                "saved", args.model_name, "history_{}.pkl".format(args.model_name)
            ),
            "wb",
        ),
    )

    save_dir = 'saved/best_epoch.csv'
    # 检查文件是否存在
    if not os.path.exists(save_dir) or os.path.getsize(save_dir) == 0:
        with open(save_dir, 'a+', newline='') as logfile:
            logwriter = csv.DictWriter(logfile,
                                       fieldnames=['lr', 'a',  'best_epoch', 'best_ddi_rate', 'best_ja', 'best_avg_f1',
                                                   'best_prauc', 'best_avg_med', 'best_avg_p', 'best_avg_r'])
            logwriter.writeheader()

    # 写入数据行
    with open(save_dir, 'a+', newline='') as logfile:
        logwriter = csv.DictWriter(logfile,
                                   fieldnames=['lr', 'a',  'best_epoch', 'best_ddi_rate', 'best_ja', 'best_avg_f1',
                                               'best_prauc', 'best_avg_med', 'best_avg_p', 'best_avg_r'])
        logdict = dict(lr=args.lr, a=args.a, best_epoch=best_epoch, best_ddi_rate=best_ddi_rate, best_ja=best_ja,
                       best_avg_f1=best_avg_f1, best_prauc=best_prauc, best_avg_med=best_avg_med, best_avg_p=best_avg_p,
                       best_avg_r=best_avg_r)
        logwriter.writerow(logdict)


if __name__ == "__main__":
    set_seed()
    main()
