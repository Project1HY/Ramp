import numpy as np
from sklearn.neighbors import NearestNeighbors
from benchmark_stats.compute_icp import *
import trimesh

def collect_reconstruction_stats(gt, part, tp, comp):

    
    N = len(gt)
    me_err = np.zeros((N,1))
    vol_err = np.zeros((N,1))
    # maybe add chamfer later
    tp_me_err = np.zeros((N,1))
    # correpondence = np.zeros((N,1))
    # correspondence_10_hitrate = np.zeros((N,1))
    #progress bar - through wandb
    for i in range(N):
        [comp, gt, part, tp] = [comp[i], gt[i], part[i], tp[i]] #TODO: load gt and completions vertices
        #GT res
        comp = icp(comp, gt, True)
        diff_tp = np.power(abs(icp(tp,gt,True) - gt), 2)
        tp_me_err[i] = (np.sum(diff_tp[:])/len(tp))
        # Compute Mean Error
        diff = np.power(abs(comp - gt),2) #MSE
        me_err[i] = np.sqrt(sum(diff[:])/len(comp))
        #Compute Volume Error
        gtvol = trimesh.Trimesh(gt) #TODO: add calc
        gtvol = gtvol.volume
        vol_err[i] = abs(gtvol - comp.volume())/gtvol
        # Align Part<->Res
        part = icp(part,comp,True)
        #_, correspondence[i] = nearest_neighbor(res_v, part_v)
        #correspondence_10_hitrate[i] = np.count_nonzero(correspondence[i]-mask)/len(mask)

    stats = {}
    stats['mean_error'] = me_err
    stats['volume_error'] = vol_err
    stats['template_mean_error'] = tp_me_err


    return stats


#TODO: report through wandb



