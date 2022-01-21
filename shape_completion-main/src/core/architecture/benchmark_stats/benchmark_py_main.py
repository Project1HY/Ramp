# Consts
from sklearn.neighbors import NearestNeighbors
from benchmark_python.private.collect_reconstruction_stats_py import *
from benchmark_python.private.calc_dist_and_Collect_geoerror import *

PATH_SORT_OPT = ['seq','highestL2','lowestL2','lowest_chamfergt2res','lowest_chamferres2gt','rand','none']
path_data_dir =  fullfile(up_script_dir(2),'data','synthetic')
path_exps_dir = fullfile(up_script_dir(0),'COMPLETIONS')
path_collat_dir = fullfile(up_script_dir(0),'collaterals')
path_tmp_dir = fullfile(c.path.collat_dir,'tmp')
[c.f,c.f_ds] = get_representative_tri(c) # Presuming same triv

% Targets
exp_targets = {'EXP16c_Faust2Faust'}
#c.exp_targets = {'EXP_Ablation_2Faust'};
#c.exp_targets = {'EXP1_SMAL2SMAL'};
#c.exp_targets = list_file_names(c.path.exps_dir);
#c.exp_targets = c.exp_targets(startsWith(c.exp_targets,'EXP'));

#Path
# sort_meth = PATH_SORT_OPT[6]
# pick_n_at_random = 20;
# c.no_self_match = 0;
# % c.look_at_sub = {'8','9'};
# % c.look_at_template_pose = {'0'};
# % c.look_at_projection_pose = {'0'};
# % c.look_at_projection_id = {'6','3','8'};
# c.export_subset = 0;

# % Geodesic Error
# c.n_run_geodesic_err = 0;
# c.geodesic_err_xlim = [0,0.2];
# c.visualize_stats = 0;

# % Visualization
# c.n_renders = 0;
# c.cherry_pick_mode = 0;
# c.export_render_to_ply = 1;
# c.write_gif = 0; c.frame_rate = 0.3;

%-------------------------------------------------------------------------%
%
%-------------------------------------------------------------------------%

for i in range(len(exp_targets)):
    banner(exp_targets[i]);
    c = tailor_config_to_exp(c,c.exp_targets[i])
    paths = #TODO: add paths
    stats, _= collect_reconstruction_stats_py(c,paths)
#     [paths,stats] = filter_subset(c,stats,paths)
#     [avg_curve,~] = collect_geoerr(c,stats,paths)
#     visualize_statistics(c,paths,stats,avg_curve)
#     visualize_results(c,stats,paths)

# banner('done');



