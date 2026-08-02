# Retargeter implementation notes (from the project owner)

1. The old ASE retargeting code (mocap_retarget.py, retarget_dog_to_*.py) does
   NOT work — the owner wrote it as an attempt and abandoned it. Treat it only
   as a record of intent (joint naming, coordinate conventions), never as a
   correctness reference. Write the retargeter fresh.
2. The poselib BVH parsing (SkeletonMotion.from_bvh, Y-up to Z-up rotation,
   skeleton3d.py) IS trustworthy — the Go2 NPYs in use today came through it.
3. The dm_control dog has far more DOF than the 21-joint BVH. Approximate:
   distribute BVH Spine+Spine1 rotation across the 7 lumbar joints (slerp
   fractions, projected per-joint onto each hinge axis), Neck/Head across the
   cervical chain + skull, Tail/Tail1 across the caudal chain. Legs and front
   limbs map closest-chain 1:1. Unmapped joints (fingers, jaw, toes) stay at
   rest pose.
4. Validate numerically at every stage; do not assume any inherited code path
   is correct.
