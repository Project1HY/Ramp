import numpy as np
from sklearn.neighbors import NearestNeighbors
from .compute_icp import *
import trimesh

def collect_reconstruction_stats(gts,masks, tps, comps,faces):

    if len(comps.shape) > 3:
        comps = comps.reshape(-1, comps.shape[-2], comps.shape[-1])
    if len(gts.shape) > 3:
        gts = gts.reshape(-1, gts.shape[-2], gts.shape[-1])
    if len(tps.shape) > 3:
        tps = tps.reshape(-1, tps.shape[-2], tps.shape[-1])

    
    N = len(gts)
    me_err = np.zeros((N,1))
    me_no_icp_err = np.zeros((N,1))
    vol_err = np.zeros((N,1))
    vol_err_pre_icp = np.zeros((N,1))
    # maybe add chamfer later
    tp_me_err = np.zeros((N,1))
    tp_me_err_pre_icp = np.zeros((N,1))

    tp_comp_me_error = np.zeros((N,1))
    tp_comp_me_error_pre_icp = np.zeros((N,1))

    # correpondence = np.zeros((N,1))
    # correspondence_10_hitrate = np.zeros((N,1))
    #progress bar - through wandb
    for i in range(N):
        [comp, gt, mask, tp] = [comps[i], gts[i], masks[i], tps[i]] #TODO: load gt and completions vertices
        #GT res
        part = gt[mask,:]
        gt = gt[:,:3]
        tp = tp[:,:3]

        comp_pre_icp = comp
        
        diff_tp_pre_icp = np.power(abs(tp - gt), 2)
        tp_me_err_pre_icp[i] = (np.sum(diff_tp_pre_icp[:])/tp.shape[0])



        diff_tp_comp_no_icp = np.power(abs(comp_pre_icp - tp), 2)
        tp_comp_me_error_pre_icp[i] = (np.sum(diff_tp_comp_no_icp[:])/tp.shape[0])
        
        # Compute Mean Error
        
        diff_no_icp = np.power(abs(comp_pre_icp-gt),2) #MSE NO ICP
        me_no_icp_err[i] = np.sqrt(np.sum(diff_no_icp)/comp.shape[0])

        #Compute Volume Error
        gtvol = trimesh.Trimesh(vertices = gt, faces=faces, process=False) #TODO: add calc
        gtvol = gtvol.volume
        
        
        compvol_pre_icp = trimesh.Trimesh(vertices = comp_pre_icp, faces=faces, process=False).volume
        vol_err_pre_icp[i] = abs(gtvol - compvol_pre_icp)/abs(gtvol)

        # Align Part<->Res
        #part = icp(part[:,:3],comp[:,:3],True)
        #_, correspondence[i] = nearest_neighbor(res_v, part_v)
        #correspondence_10_hitrate[i] = np.count_nonzero(correspondence[i]-mask)/len(mask)

    stats = {}
    stats['Comp-GT Vertex L2'] = list(me_no_icp_err[:,0])
    stats['Comp-GT Volume L1'] = list(vol_err_pre_icp[:,0])
    stats['TP-GT Vertex L2'] = list(tp_me_err_pre_icp[:,0])
    stats['Comp-TP Vertex L2'] = list(tp_comp_me_error_pre_icp[:,0])

    return stats

#TODO: report through wandb



