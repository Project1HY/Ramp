from scipy.sparse.linalg import splu
from geom.matrix.cpu import last_axis_normalize, last_axis_2norm
from geom.mesh.op.cpu.spectra import laplacian, barycenter_vertex_mass_matrix
from geom.mesh.io.base import read_mesh
from cfg import TEST_MESH_HUMAN_PATH
from geom.mesh.vis.base import add_isocontours, add_mesh, add_spheres
from geom.mesh.op.cpu.base import surface_area, face_areas
from pathlib import Path
from scipy import sparse
import pyvista as pv
import gdist
from gdist import compute_gdist as ExactGeodesic
import torch
import numpy as np
import multiprocessing as mp
from util.time import progress


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#
def geodesic_distances(v, f, src=None, dest=None, normalize_by_area=True,
                       max_distance=float('inf'), num_workers=0):
    """
    :param v: [nv x 3] vertices
    :param f: [nf x 3] faces
    :param src: np.array of source nodes or None to use all
    :param dest: np.array of destination nodes or None to use all
    :param normalize_by_area: Normalizes by sqrt(mesh_surface_area)
    :param max_distance: If supplied, will only yield results for geodesic distances with less than max_distance.
    This will speed up runtime dramatically.
    :param num_workers: How many subprocesses to use for calculating geodesic distances.
        num_workers = 0 means that computation takes place in the main process.
        num_workers = -1 means that the available amount of CPU cores is used.
    :return: 
    """

    area_normalization = np.sqrt(surface_area(v,f)) if normalize_by_area else 1


    if src is None and dest is None:
        out = gdist.local_gdist_matrix(v, f, max_distance * area_normalization) / area_normalization
    if src is None:
        src = np.arange(v.shape[0], dtype=np.int32)

    dest = None if dest is None else dest

    num_workers = mp.cpu_count() if num_workers <= -1 else num_workers
    if num_workers > 0:
        with mp.Pool(num_workers) as pool:
            outs = pool.starmap(
                _parallel_loop,
                [(v, f, src, dest, max_distance, normalize_by_area, i)
                 for i in range(len(src))])
    else:
        outs = [
            _parallel_loop(v, f, src, dest, max_distance, normalize_by_area, i)
            for i in range(len(src))
        ]

    return np.cat(outs, dim=0)


def _parallel_loop(pos, face, src, dest, max_distance, norm, i):
    s = src[i:i + 1]
    d = None if dest is None else dest[i:i + 1]
    return gdist.compute_gdist(pos, face, s, d, max_distance * norm) / norm



class HeatMethod:
    def __init__(self, v, f, m=10.0):
        self.v, self.f = v, f

        # Precomputation of all reusables:
        # Edge Vectors:
        e01 = v[f[:, 1]] - v[f[:, 0]]
        e12 = v[f[:, 2]] - v[f[:, 1]]
        e20 = v[f[:, 0]] - v[f[:, 2]]
        # Face Areas
        self.fa = .5 * last_axis_2norm(np.cross(e01, e12))
        # Edge Normals:
        face_normals = last_axis_normalize(np.cross(last_axis_normalize(e01), last_axis_normalize(e12)))
        self.e1_normal = np.cross(face_normals, e01)
        self.e2_normal = np.cross(face_normals, e12)
        self.e3_normal = np.cross(face_normals, e20)
        # h Parameter Heuristic - Mean Edge Length
        h = np.mean(list(map(last_axis_2norm, [e01, e12, e20])))
        t = m * h ** 2
        # pre-factorize poisson systems
        L = laplacian(v, f, cls='meyer')
        A = barycenter_vertex_mass_matrix(v, f)
        self.lu_decomposed_area_laplacian_func = splu((A - t * L).tocsc()).solve
        self.lu_decomposed_laplacian_func = splu(L.tocsc()).solve

    def __call__(self, vi):
        """
        computes geodesic distances to all vertices in the mesh
        idx can be either an integer (single vertex index) or a list of vertex indices
        or an array of bools of length n (with n the number of vertices in the mesh)
        """
        u0 = np.zeros(len(self.v))
        u0[vi] = 1.0

        # Step 1:
        u = self.lu_decomposed_area_laplacian_func(u0).ravel()
        # Step 2:
        grad_u = 1 / (2 * self.fa)[:, np.newaxis] * (
                self.e1_normal * u[self.f[:, 2]][:, np.newaxis]
                + self.e2_normal * u[self.f[:, 0]][:, np.newaxis]
                + self.e3_normal * u[self.f[:, 1]][:, np.newaxis]
        )
        X = - grad_u / last_axis_2norm(grad_u)[:, np.newaxis]
        # Step 3:
        div_Xs = np.zeros(len(self.v))
        for i1, i2, i3 in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:  # for edge i2 --> i3 facing vertex i1
            vi1, vi2, vi3 = self.f[:, i1], self.f[:, i2], self.f[:, i3]
            e1 = self.v[vi2] - self.v[vi1]
            e2 = self.v[vi3] - self.v[vi1]
            e_opp = self.v[vi3] - self.v[vi2]
            cot1 = 1 / np.tan(np.arccos(
                (last_axis_normalize(-e2) * last_axis_normalize(-e_opp)).sum(axis=1)))
            cot2 = 1 / np.tan(np.arccos(
                (last_axis_normalize(-e1) * last_axis_normalize(e_opp)).sum(axis=1)))
            div_Xs += np.bincount(
                vi1.astype(int),
                0.5 * (cot1 * (e1 * X).sum(axis=1) + cot2 * (e2 * X).sum(axis=1)),
                minlength=len(self.v))
        phi = self.lu_decomposed_laplacian_func(div_Xs).ravel()
        phi -= phi.min()
        return phi


# ---------------------------------------------------------------------------------------------------------------------#
#                                      # TODO - waiting for constraint matrix support
# ---------------------------------------------------------------------------------------------------------------------#
def main():
    v, f = read_mesh(TEST_MESH_HUMAN_PATH)
    # Choose Source:
    sources = [v[:, 2].argmax(), 5000]  # Take a radical vertex in z as a source

    dist = HeatMethod(v, f, 10.0)(sources)
    true_dist = ExactGeodesic(v, f, np.array(sources, dtype=np.int32))
    # assert np.allclose(dist, dist2)

    # # Plot:
    p = pv.Plotter(shape=(1, 2))
    p.subplot(0, 0)
    add_mesh(p, v, f, clr=dist, cmap='Blues', smooth_shade_on=True, strategy='mesh')
    add_spheres(p, v[sources])
    add_isocontours(p, v, f, scalar_func=dist)

    p.subplot(0, 1)
    add_mesh(p, v, f, clr=true_dist, cmap='Blues', smooth_shade_on=True, strategy='mesh')
    add_spheres(p, v[sources])
    add_isocontours(p, v, f, scalar_func=true_dist)
    p.link_views()
    p.show()



if __name__ == '__main__':
    main()

# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#
# function [edge_plots] = add_geodesic_paths(M,paths,clrs)
#
# if ~exist('clrs','var'); clrs = 'c'; end
# if ~iscell(paths)
#     paths = {paths};
# end
#
# edge_plots = zeros(1,numel(paths)); %For legend
# hold on;
# for i=1:numel(paths)
#     P = paths{i};
#     if iscell(clrs)
#         clr = clrs{i};
#     else
#         clr = clrs;
#     end
#     if size(P,1)==1 % Vertices:
#         add_feature_pts(M,[M.v(P(1),:);M.v(P(end),:)]);
#         E =[P(1:end-1);P(2:end)].';
#         AP = add_edge_visualization(M,E,0,clr);
#         edge_plots(i) = AP(1);
#     else % Geodesic
#         add_feature_pts(M,[P(:,1),P(:,end)].');
#         hold on;
#         AP = plot3(P(1,:), P(2,:),P(3,:), clr,'LineWidth',2);
#         hold off;
#         edge_plots(i) = AP(1);
#     end
# end
# hold off;

# clearvars; close all; clc; demos_setup(); cd(up_script_dir(0));
# %-------------------------------------------------------------------------%
# %
# %-------------------------------------------------------------------------%
# [M,libm ,menu] = test_mesh(3,'small',0);
#
#  [~,s] = max(M.y());
#  t = mesh_FPS(M,2,s); t = t(2);
#
#
# [P_graph,len_graph] = mesh_path(M,s,t,'graph');
# [P_euclid_graph,len_egraph] = mesh_path(M,s,t,'euclidean-graph');
# [P_geodiscrete,len_geodisc] = mesh_path(M,s,t,'geodesic-discrete');
# [P_geo,len_geo] = mesh_path(M,s,t,'geodesic');
#
# M.plt.title = usprintf('Mesh Paths on %s',M.name);
# M.ezvisualize();
# leg = {sprintf('Graph Path [%g]',len_graph),...
#     sprintf('Euclidean Graph Path [%g]',len_egraph),...
#     sprintf('Geodesic Discrete Path[%g]',len_geodisc),...
#     sprintf('Geodesic Path [%g]',len_geo)};
# EP = add_geodesic_paths(M,{P_graph,P_euclid_graph,P_geodiscrete,P_geo},{'r','c','m','g'});
# legend(EP,leg)

# function [D] = mesh_distance_matrix(M,method)
#
# if ~exist('method','var'); method = 'geodesic'; end
# switch lower(method)
#     case 'geodesic'
#         D = mesh_fmm_mat(M);
#     case 'graph'
#         G = graph(M.A);
#         D = distances(G);
#     case 'euclidean-graph'
#         [De,E] = M.edge_distances();
#         W = sparse([E(:,1);E(:,2)],[E(:,2);E(:,1)],[De;De],M.Nv,M.Nv);
#         G = graph(W);
#         D = distances(G);
#     otherwise
#         error('Unimplemented distance %s',method);
# end
# end
#
# function D = mesh_fmm_mat(M)
#
# W = ones(M.Nv,1);
# dmax = 1e9; iter_max = 1.2*M.Nv;
# D =zeros(M.Nv);
#
# v = M.v.'; f = M.f.'-1;
# progressbar;
# for i=1:M.Nv
#     D(:,i) = perform_front_propagation_mesh(v,f,W,i-1,[],iter_max, [], [], [], dmax);
#     progressbar(i/M.Nv);
# end
# D(D>dmax)= Inf;
# end

# function P = add_edge_visualization(M,E,is_fedge,clr)
#
# if ~exist('is_fedge','var'); is_fedge = 0; end
# if ~exist('clr','var') || isempty(clr); clr = 'c'; end
#
# E = E.';
# hold on;
# if is_fedge
#    M = M.add_face_centers();
#     x = M.fc(:,1); y = M.fc(:,2); z = M.fc(:,3);
# %      M = M.add_face_normals();
# %     x = M.fc(:,1)+0.01*M.fn(:,1); y = M.fc(:,2)+0.01*M.fn(:,2); z = M.fc(:,3)+0.01*M.fn(:,3);
# else
#    x = M.v(:,1); y = M.v(:,2); z = M.v(:,3);
# end
# P = plot3(x(E), y(E), z(E), 'Color',clr, 'LineWidth',1.5);
# hold off;
# end


# function [P,len] = mesh_path(M,s,t,method)
# if ~exist('method','var'); method = 'geodesic'; end
# switch lower(method)
#     case 'geodesic'
#         [P,len] = geodesic_path_extraction(M,s,t);
#     case 'geodesic-discrete'
#         [P,len] = discrete_geodesic_path_extraction(M,s,t);
#     case 'graph'
#         G = graph(M.A);
#         [P,len] = shortestpath(G,s,t);
#     case 'euclidean-graph'
#         [De,E] = M.edge_distances();
#         W = sparse([E(:,1);E(:,2)],[E(:,2);E(:,1)],[De;De],M.Nv,M.Nv);
#         G = graph(W);
#         [P,len] = shortestpath(G,s,t);
#     otherwise
#         error('Unimplemented distance %s',method);
# end
# end
# pygraphviz‑1.3.1‑cp34‑none‑win_amd64.whl
# function [P,len] = discrete_geodesic_path_extraction(M,s,t)
#
# % Compute the distance vector
# opt.t = t;
# [D] = mesh_fast_marching(M,s,opt);
# % Init
# M = M.add_vertex_ring();
# P = t;
# curr_dist = D(t);
# if curr_dist == Inf
#     warning('Failed to find path from s -> t');
#     len = Inf;
#     return
# end
# len = 0;
# % Dijakstra:
# while curr_dist ~= 0
#     % Find closest neighbor to target
#     curr_v = P(end);
#     neighs = M.Rv{curr_v};
#     [next_dist,I] = min(D(neighs));
#     % Validate this neighbor helps us
#     if next_dist >= curr_dist
#         warning('Failed to find path from s -> t');
#         break;
#     end
#     % Add neighbor to path:
#     P(end+1) = neighs(I);
#     len = len + normv(M.v(curr_v,:) - M.v(neighs(I),:));
#     curr_dist = next_dist;
# end
#
# end
#
# function [P,len] =geodesic_path_extraction(M,s,t)
# opt.t = t;
# [D] = mesh_fast_marching(M,s,opt);
# M = M.add_vertex_ring();
# len = D(t);
# [P,~,~] = compute_geodesic_mesh(D,M.v.',M.f.',t,M.Rv); % TODO: Merge this
# end
#
# function [path,vlist,plist] = compute_geodesic_mesh(D, vertex, face, x,Rv)
#
# %%% gradient descent on edges
# % precompute the adjacency datasets
# e2f = compute_edge_face_ring(face);
# % initialize the paths
# [w,f] = vertex_stepping(face, x, e2f, Rv, D);
#
# vlist = [x;w];
# plist = 1;
# Dprev = D(x);
#
# while true
#     % current triangle
#     i = vlist(1,end);
#     j = vlist(2,end);
#     k = get_vertex_face(face(:,f),i,j);
#     a = D(i); b = D(j); c = D(k);
#     % adjacent faces
#     f1 = get_face_face(e2f, f, i,k);
#     f2 = get_face_face(e2f, f, j,k);
#     % compute gradient in local coordinates
#     x = plist(end); y = 1-x;
#     gr = [a-c;b-c];
#     % twist the gradient
#     u = vertex(:,i) - vertex(:,k);
#     v = vertex(:,j) - vertex(:,k);
#     A = [u v]; A = (A'*A)^(-1);
#     gr = A*gr;
#     nx = gr(1); ny = gr(2);
#     % compute intersection point
#     etas = -y/ny;
#     etat = -x/nx;
#     s = x + etas*nx;
#     t = y + etat*ny;
#     if etas<0 && s>=0 && s<=1 && f1>0
#         %%% CASE 1 %%%
#         plist(end+1) = s;
#         vlist(:,end+1) = [i k];
#         % next face
#         f = f1;
#     elseif etat<0 && t>=0 && t<=1 && f2>0
#         %%% CASE 2 %%%
#         plist(end+1) = t;
#         vlist(:,end+1) = [j k];
#         % next face
#         f = f2;
#     else
#         %%% CASE 3 %%%
#         if a<=b
#             z = i;
#         else
#             z = j;
#         end
#         [w,f] = vertex_stepping( face, z, e2f, Rv, D);
#         vlist(:,end+1) = [z w];
#         plist(end+1) = 1;
#     end
#     Dnew = D(vlist(1,end))*plist(end) + D(vlist(2,end))*(1-plist(end));
#     if Dnew==0 || (Dprev==Dnew && length(plist)>1)
#         break;
#     end
#     Dprev=Dnew;
# end
#
# v1 = vertex(:,vlist(1,:));
# v2 = vertex(:,vlist(2,:));
#
# path = v1.*repmat(plist, [3 1]) + v2.*repmat(1-plist, [3 1]);
# end
#
# function [w,f] = vertex_stepping(face, v, e2f, v2v, D)
#
# % adjacent vertex with minimum distance
# [~,I] = min( D(v2v{v}) ); w = v2v{v}(I);
# f1 = e2f(v,w);
# f2 = e2f(w,v);
# if f1<0
#     f = f2; return;
# end
# if f2<0
#     f = f1; return;
# end
# z1 = get_vertex_face(face(:,f1),v,w);
# z2 = get_vertex_face(face(:,f2),v,w);
# if D(z1)<D(z2)
#     f = f1;
# else
#     f = f2;
# end
# end
#
# function k = get_vertex_face(f,v1,v2)
#
# if nargin==2
#     v2 = v1(2); v1 = v1(1);
# end
# k = setdiff(f, [v1 v2]);
# if length(k)~=1
#     error('Error in get_vertex_face');
# end
# end
#
# function f = get_face_face(e2f, f, i,j)
#
# f1 = e2f(i,j); f2 = e2f(j,i);
# if f==f1
#     f = f2;
# else
#     f = f1;
# end
# end
# function A = compute_edge_face_ring(face)
#
# % compute_edge_face_ring - compute faces adjacent to each edge
# %
# %   e2f = compute_edge_face_ring(face);
# %
# %   e2f(i,j) and e2f(j,i) are the number of the two faces adjacent to
# %   edge (i,j).
# %
# %   Copyright (c) 2007 Gabriel Peyre
# n = max(face(:));
# m = size(face,2);
# i = [face(1,:) face(2,:) face(3,:)];
# j = [face(2,:) face(3,:) face(1,:)];
# s = [1:m 1:m 1:m];
#
# % first without duplicate
# [~,I] = unique( i+(max(i)+1)*j );
# % remaining items
# J = setdiff(1:length(s), I);
#
# % flip the duplicates
# i1 = [i(I) j(J)];
# j1 = [j(I) i(J)];
# s = [s(I) s(J)];
#
# % remove doublons
# [~,I] = unique( i1+(max(i1)+1)*j1 );
# i1 = i1(I); j1 = j1(I); s = s(I);
#
# A = sparse(i1,j1,s,n,n);
#
#
# % add missing points
# I = find( A'~=0 );
# I = I( A(I)==0 );
# A( I ) = -1;
# end
#
#
