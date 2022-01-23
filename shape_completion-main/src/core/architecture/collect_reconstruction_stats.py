import numpy as np
from sklearn.neighbors import NearestNeighbors
from .compute_icp import *
import trimesh

def collect_reconstruction_stats(gts,masks, tps, comps,faces):

    
    N = len(gts)
    me_err = np.zeros((N,1))
    vol_err = np.zeros((N,1))
    # maybe add chamfer later
    tp_me_err = np.zeros((N,1))
    # correpondence = np.zeros((N,1))
    # correspondence_10_hitrate = np.zeros((N,1))
    #progress bar - through wandb
    for i in range(N):
        [comp, gt, mask, tp] = [comps[i], gts[i], masks[i], tps[i]] #TODO: load gt and completions vertices
        #GT res
        part = gt[mask,:]
        gt = gt[:,:3]
        tp = tp[:,:3]
        gt = gt.numpy()
        tp = tp.numpy()
        comp = icp(comp, gt, True)
        diff_tp = np.power(abs(icp(tp,gt,True) - gt), 2)
        tp_me_err[i] = (np.sum(diff_tp[:])/tp.shape[0])
        # Compute Mean Error
        diff = np.power(abs(comp - gt),2) #MSE
        me_err[i] = np.sqrt(np.sum(diff)/comp.shape[0])
        #Compute Volume Error
        gtvol = trimesh.Trimesh(vertices = gt, faces=faces, process=False) #TODO: add calc
        gtvol = gtvol.volume
        compvol = trimesh.Trimesh(vertices = comp, faces=faces, process=False).volume
        vol_err[i] = abs(gtvol - compvol)/gtvol
        # Align Part<->Res
        #part = icp(part[:,:3],comp[:,:3],True)
        #_, correspondence[i] = nearest_neighbor(res_v, part_v)
        #correspondence_10_hitrate[i] = np.count_nonzero(correspondence[i]-mask)/len(mask)

    stats = {}
    stats['mean_error'] = list(me_err[:,0])
    stats['volume_error'] = list(vol_err[:,0])
    stats['template_mean_error'] = list(tp_me_err[:,0])
    

    #best = {}
    #best['mean_error'] = min(stats['mean_error'])
    #best['volume_error'] = min(stats['volume_error'])
    #best['template_mean_error'] = min(stats['template_mean_error'])
    #best_mean_index = stats['mean_error']).index(min(stats['mean_error']))
    #best_tp_by_mean = tps[best_mean_index]

    #return stats, best
    return stats

#TODO: report through wandb



