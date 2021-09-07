import os
import bpy
import os


subject = '000'
load_dir_base = 'D:/Data/Mixamo/website_downloads/MPI-FAUST/' 
save_dir_base = 'D:\Data/Mixamo/Blender/MPI-FAUST/'


load_dir_subject = os.path.join(load_dir_base, subject)
save_dir_subject = os.path.join(save_dir_base, subject)
if os.path.exists(save_dir_subject) == False:
    os.mkdir(save_dir_subject)

for filename in os.listdir(load_dir_subject):
    print(filename)
    if filename.endswith(".fbx"): 
        path_animation = os.path.join(load_dir_subject, filename)
        motion = os.path.splitext(filename)[0]
        save_dir_motion = os.path.join(save_dir_subject, motion) 
        if os.path.exists(save_dir_motion) == False:
            os.mkdir(save_dir_motion)
        path_save = os.path.join(save_dir_motion, '%03d.obj')
        
        #clean scene
        for obj in bpy.data.objects:
            bpy.data.objects.remove(obj)
       
        for mesh in bpy.data.meshes:
            bpy.data.meshes.remove(mesh)
            
        #load motion sequence
        bpy.ops.import_scene.fbx(filepath = path_animation)
        N = int(bpy.data.objects['Armature'].animation_data.action.frame_range[1])
        for i in range(N):
            bpy.context.scene.frame_set(i + 1)
            bpy.ops.export_scene.obj(filepath=path_save % (i + 1))
