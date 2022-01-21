import skfmm
import numpy as np

def calc_dist_mat(v,f):
    nv = len(v)
    march = skfmm.distance(v)
    D = np.zeros(nv,nv)
    for i in range(len(nv)):
        source = np.ones(nv,1)*np.inf
        source[i] = 0
        d = skfmm.distance(float(source))
        D[:,i] = d[:]
    skfmm.distance(march)
    D = 0.5*(D+D.transpose()) #what does it mean?


def collect_geoerror(c, stats, paths):

# function [avg_curve,curves] = collect_geoerr(c,stats,paths)

# N = min(size(paths,1),c.n_run_geodesic_err);
# % if N == 0; return; end

# curves = zeros(N,1001);
# corr = stats.correspondence;
# % progressbar;
# if N>0
#     ppm = ParforProgressbar(N);
#     parfor i=1:N
        
#         [~,~,~,tempM,mask] = load_path_tup(c,paths(i,:));
#         D = calc_dist_matrix(tempM.v,tempM.f);
#         curves(i,:) = calc_geo_err(corr{i},mask, D);
#         ppm.increment();
#         % progressbar(i/N);
#     end
#     delete(ppm);
# end

# avg_curve = sum(curves,1)/ size(curves,1);
# end

# % matches_refined = refine_correspondence(partM.v,partM.f,tempM.v,tempM.f,matches_reg);