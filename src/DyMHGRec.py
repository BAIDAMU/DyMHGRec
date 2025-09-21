import torch
import argparse
import numpy as np
import dill
import time
from torch.optim import Adam
import os
import torch.nn.functional as F
from collections import defaultdict
from scipy.sparse import csr_matrix
from Hyp_models_monmery_dim32 import HypDrug
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params
import math
torch.manual_seed(1203)
np.random.seed(1203)

model_name = 'DyMHGRec'
resume_name = ''

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--eval', action='store_true', default=True, help="eval mode")
parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--resume_path', type=str, default=resume_name, help='resume path')

args = parser.parse_args()
model_name = args.model_name
resume_name = args.resume_path

def eval(model, data_eval, divided_val, voc_size, epoch):
    model.eval()
    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    med_cnt, visit_cnt = 0, 0
    case_study = defaultdict(dict)
    
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
        current_ddi_rate = ddi_rate_score([[y_pred_label_tmp]])
        real_ddi = ddi_rate_score([[sorted(adm[2])]])
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
        )
        case_study[step] = {'ja': adm_ja, 'patient': input, 'y_label': y_pred_label, 'output':y_pred_prob, 'ddi_rate': current_ddi_rate, 'real_ddi':real_ddi}
        
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
    
    dill.dump(case_study, open(os.path.join('saved', model_name, 'DyMHGRec1.pkl'), 'wb'))
    return (
        ddi_rate,
        np.mean(ja),
        np.mean(prauc),
        np.mean(avg_p),
        np.mean(avg_r),
        np.mean(avg_f1),
        med_cnt / visit_cnt
    )



def main():
    if not os.path.exists(os.path.join("saved", model_name)):
        os.makedirs(os.path.join("saved", model_name))

     # load data
    data_path = "../data/data_new/records_final_131.pkl"
    # TODO: Shuffule-data131
    # data_path = "../random_bigdata131_5/data/records_final_131_shuffle5.pkl"
    voc_path = "../data/data_new/voc_final.pkl"


    ddi_adj_path = "../data/data_new/ddi_A_final.pkl"
    ddi_mask_path = "../data/data_new/ddi_mask_H.pkl"
    molecule_path = "../data/data_new/idx2drug.pkl"

    ddi_A_final_path = "../data/data_new/ddi_A_final.pkl"
    # TODO :ehr
    ehr_adj_path = '../data/data_new/ehr_adj_final.pkl'
    pretrained_path = "../pretrain/pretrained_model.pt"
    adj_diag_med_path = "../data/data_new/adj_diag_med.pkl"

    #数据分类
    all_divided_path = "../data/data_divided/all_divided.pkl"


    device = torch.device('cuda:0')

    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    ddi_adj = dill.load(open(ddi_adj_path, "rb"))
    ddi_mask_H = dill.load(open(ddi_mask_path, "rb"))
    data = dill.load(open(data_path, "rb"))
    molecule = dill.load(open(molecule_path, "rb"))
    adj_diag_med = dill.load(open(adj_diag_med_path, "rb"))
    
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

    TEST = args.eval


    model = HypDrug(
        voc_size,
        ehr_adj,
        ddi_adj,
        ddi_mask_H,

        emb_dim=128,
        device=device,
    )
   
    if TEST:
        model.load_state_dict(torch.load(open('saved/Hyp_monmery_dim128/Epoch_63_TARGET_0.06_JA_0.6815_DDI_0.04874.model', 'rb')))
    model.to(device=device)

    print('parameters', get_n_params(model))

    if TEST:
        eval(model, data_eval, eval_divided, voc_size, 0)
    else:
       
        print('best_epoch:', "error")


if __name__ == '__main__':
    main()
