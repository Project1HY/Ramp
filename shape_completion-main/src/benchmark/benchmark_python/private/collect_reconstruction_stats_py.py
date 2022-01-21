
import numpy as np
from sklearn.neighbors import NearestNeighbors
from benchmark_python.private.compute_icp import *

def collect_reconstruction_stats_py(c, paths, force):

    if force: #TODO: or other conditions
        N = len(paths)
        me_err = np.zeros((N,1))
        vol_err = np.zeros((N,1))
        # maybe add chamfer later
        tp_me_err = np.zeros((N,1))
        correpondence = np.zeros((N,1))
        correspondence_10_hitrate = np.zeros((N,1))
        #progress bar - through wandb
        for i in range(N):
            [res, gt, part, tp, mask] = load_paths(c, paths[i,:]) #TODO: load gt and completions vertices
            #GT res
            res_v = icp(res_v, gt_v, True)
            diff_tp = np.power(abs(compute_icp(tp_v,gt_v,True) - gt_v), 2)
            tp_me_err[i] = (np.sum(diff_tp[:])/len(tp_v))
            # Compute Mean Error
            diff = np.power(abs(res_v - gt_v),2) #MSE
            me_err[i] = np.sqrt(sum(diff[:])/len(res_v))
            #Compute Volume Error
            gtvol = gt_volume #TODO: add calc
            vol_err[i] = abs(gtvol - res.volume())/gtvol
            # Align Part<->Res
            part_v = icp(part_v,res_v,True)
            _, correspondence[i] = nearest_neighbor(res_v, part_v)
            #correspondence_10_hitrate[i] = np.count_nonzero(correspondence[i]-mask)/len(mask)


    return stats, curr_stats



    # stats.(c.curr_exp) = struct;
    # stats.(c.curr_exp).me_err = me_err;
    # stats.(c.curr_exp).chamfer_gt2res = chamfer_gt2res;
    # stats.(c.curr_exp).chamfer_res2gt = chamfer_res2gt;
    # stats.(c.curr_exp).vol_err = vol_err;
    # stats.(c.curr_exp).correspondence = correspondence;
    # stats.(c.curr_exp).correspondence_10_hitrate = correspondence_10_hitrate;
    # stats.(c.curr_exp).tp_me_err = tp_me_err; 
    
    
    save(tgt_stats_fp,'stats');

#TODO: report through wandb



