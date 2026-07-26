# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Merge a multi-part UE FBX character export onto one skeleton and emit a
# textured UsdSkel asset for the overlay system. Edit TEXDIR/BASECOLOR_SUFFIX
# /OUT at top. Run: blender --background --python this_file.py
TEXDIR = "/home/bizon/sparkpack/SamuraiExport/Content/RedSamurai/Textures/Game/RedSamurai/Textures"
BASECOLOR_SUFFIX = "BaseColorGray"
OUT = "/home/bizon/sparkpack/ProtoMotions/protomotions/data/assets/overlay/gray_samurai.usd"
import bpy, glob, os, re
bpy.ops.wm.read_factory_settings(use_empty=True)
fbxs=sorted(glob.glob('/home/bizon/sparkpack/ProtoMotions/protomotions/data/assets/mesh/red_samurai/SKM_RS_*.FBX'))
base_arm=None
for f in fbxs:
    pre=set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=f)
    new=[o for o in bpy.data.objects if o not in pre]
    arms=[o for o in new if o.type=='ARMATURE']
    if base_arm is None: base_arm=arms[0]
    else:
        for ob in new:
            if ob.type=='MESH':
                for mod in ob.modifiers:
                    if mod.type=='ARMATURE': mod.object=base_arm
                ob.parent=base_arm
        for a in arms: bpy.data.objects.remove(a, do_unlink=True)
stems={s.lower(): s for s in ['Arm','Helmet','Katana_A','Katana_B','Leg','Mask','Naginata','Shirt','Shoulders','Skirt','Torso']}
def stem_for(matname):
    n=re.sub(r'^(MI_|MM_|M_)?RS_','',matname,flags=re.I).lower()
    n=re.sub(r'\.\d+$','',n)
    if n in stems: return stems[n]
    for k,v in stems.items():
        if k in n or n in k: return v
    return None
wired=0
for mat in bpy.data.materials:
    st=stem_for(mat.name)
    if not st: continue
    mat.use_nodes=True; nt=mat.node_tree
    bsdf=next((n for n in nt.nodes if n.type=='BSDF_PRINCIPLED'),None)
    if bsdf is None: continue
    def img(name, noncolor=False):
        p=os.path.join(TEXDIR, f'T_RS_{name}.PNG')
        if not os.path.exists(p): return None
        im=bpy.data.images.load(p, check_existing=True)
        if noncolor: im.colorspace_settings.name='Non-Color'
        node=nt.nodes.new('ShaderNodeTexImage'); node.image=im
        return node
    bc=img(f'{st}_{BASECOLOR_SUFFIX}')
    if bc: nt.links.new(bc.outputs['Color'], bsdf.inputs['Base Color'])
    nm=img(f'{st}_Normal', True)
    if nm:
        nmap=nt.nodes.new('ShaderNodeNormalMap')
        nt.links.new(nm.outputs['Color'], nmap.inputs['Color'])
        nt.links.new(nmap.outputs['Normal'], bsdf.inputs['Normal'])
    orm=img(f'{st}_OcclusionRoughnessMetallic', True)
    if orm:
        sep=nt.nodes.new('ShaderNodeSeparateColor')
        nt.links.new(orm.outputs['Color'], sep.inputs['Color'])
        nt.links.new(sep.outputs['Green'], bsdf.inputs['Roughness'])
        nt.links.new(sep.outputs['Blue'], bsdf.inputs['Metallic'])
    wired+=1
print('MATERIALS WIRED:', wired)
bpy.ops.wm.usd_export(filepath=OUT, export_armatures=True, export_materials=True, export_textures=True)
print('EXPORT-DONE')
