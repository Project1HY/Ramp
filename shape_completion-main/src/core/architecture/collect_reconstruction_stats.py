import numpy as np
from sklearn.neighbors import NearestNeighbors
from .compute_icp import *
import trimesh

def collect_reconstruction_stats(gts,masks, tps, comps,faces):

    
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
        gt = gt.cpu().numpy()
        tp = tp.cpu().numpy()

        comp_pre_icp = comp
        comp = icp(comp, gt, True)
        
        diff_tp = np.power(abs(icp(tp,gt,True) - gt), 2)
        tp_me_err[i] = (np.sum(diff_tp[:])/tp.shape[0])
        diff_tp_pre_icp = np.power(abs(tp - gt), 2)
        tp_me_err_pre_icp[i] = (np.sum(diff_tp_pre_icp[:])/tp.shape[0])


        diff_tp_comp = np.power(abs(icp(comp_pre_icp,tp,True) - tp), 2)
        tp_comp_me_error [i] = (np.sum(diff_tp_comp[:])/tp.shape[0])

        diff_tp_comp_no_icp = np.power(abs(comp_pre_icp - tp), 2)
        tp_comp_me_error_pre_icp[i] = (np.sum(diff_tp_comp_no_icp[:])/tp.shape[0])
        
        # Compute Mean Error
        diff = np.power(abs(comp - gt),2) #MSE
        me_err[i] = np.sqrt(np.sum(diff)/comp.shape[0])
        
        diff_no_icp = np.power(abs(comp_pre_icp-gt),2) #MSE ICP
        me_no_icp_err[i] = np.sqrt(np.sum(diff_no_icp)/comp.shape[0])

        #Compute Volume Error
        gtvol = trimesh.Trimesh(vertices = gt, faces=faces, process=False) #TODO: add calc
        gtvol = gtvol.volume
        
        compvol = trimesh.Trimesh(vertices = comp, faces=faces, process=False).volume
        vol_err[i] = abs(gtvol - compvol)/gtvol
        
        compvol_pre_icp = trimesh.Trimesh(vertices = comp_pre_icp, faces=faces, process=False).volume
        vol_err_pre_icp[i] = abs(gtvol - compvol_pre_icp)/gtvol

        # Align Part<->Res
        #part = icp(part[:,:3],comp[:,:3],True)
        #_, correspondence[i] = nearest_neighbor(res_v, part_v)
        #correspondence_10_hitrate[i] = np.count_nonzero(correspondence[i]-mask)/len(mask)

    stats = {}
    stats['Comp-GT Vertex L2'] = list(me_err[:,0])
    stats['Comp-GT Vertex L2 No ICP'] = list(me_no_icp_err[:,0])
    stats['Comp-GT Volume L1'] = list(vol_err[:,0])
    stats['Comp-GT Volume L1 No ICP'] = list(vol_err_pre_icp[:,0])

    stats['TP-GT Vertex L2'] = list(tp_me_err[:,0])
    stats['TP-GT Vertex L2 No ICP'] = list(tp_me_err_pre_icp[:,0])
    
    stats['Comp-TP Vertex L2'] = list(tp_comp_me_error[:,0])
    stats['Comp-TP Vertex L2 No ICP'] = list(tp_comp_me_error_pre_icp[:,0])

    #best = {}
    #best['mean_error'] = min(stats['mean_error'])
    #best['volume_error'] = min(stats['volume_error'])
    #best['template_mean_error'] = min(stats['template_mean_error'])
    #best_mean_index = stats['mean_error']).index(min(stats['mean_error']))
    #best_tp_by_mean = tps[best_mean_index]

    #return stats, best
    return stats

#TODO: report through wandb



