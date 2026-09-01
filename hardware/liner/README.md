# Microduck compliant beak liner reference

This two-piece insert reproduces the retention geometry used by the simulator:
a soft upper/lower pad, a rounded front lip, a rear stop, and side cheeks around
a 24 mm ball. The lower bill opens and clears the lip at release.

## Important fit limitation

The contact pocket is dimensionally matched to the simulation, not to a
verified scan of the tester's physical beak. The STL files are therefore
**reference prints, not guaranteed mounting parts**. Before a loaded test,
measure the clear inner width and usable depth of the actual upper and lower
bill, edit `beak_liner_reference.scad`, and verify that the mouth can travel
from +20 degrees to +30 degrees without binding.

Default dimensions:

- projectile: 24 mm diameter, at most 3 g;
- clear pocket width: 28 mm;
- rear-stop-to-front-lip depth: 26 mm;
- compliant pad: 1.5 mm;
- front lip: 2 mm radius;
- rear stop and side cheeks: 5 mm high;
- overall reference width: 32 mm.

## Printing

- Material: TPU 95A or softer TPU/TPE with repeatable dimensions.
- Layer height: 0.15–0.20 mm.
- Perimeters: 3.
- Infill: 20–35%; avoid a fully rigid print.
- Print each part flat, contact-pocket side upward.
- Do not use PLA, PETG, resin, or another hard/brittle material for the first
  loaded trial.

The canonical editable model is `beak_liner_reference.scad`. The included STLs
are generated from its default dimensions by `generate_reference_stl.py` using
a 0.25 mm boolean voxel union. They are single watertight manifold components;
rendering the OpenSCAD directly gives smoother circular lips.

## Fit and retention check

1. With torque off, dry-fit both pieces and confirm that no edge touches the
   jaw linkage, camera, shell, or mouth hinge throughout the complete range.
2. Use a removable, full-surface attachment appropriate for the actual shell.
   No adhesive specification is possible without identifying its material.
3. Close only to the trained +20-degree hold angle. The ball should survive
   gentle hand rotations without being crushed or visibly dented.
4. Move the lower bill slowly to +30 degrees. The ball must leave without
   catching the rear stop or either cheek.
5. Mark the insert position. Stop if either piece shifts, peels, or changes the
   mouth range.

The first powered throw remains empty-beak. Only add a lightweight 24 mm foam
ball after the empty motion and the unpowered retention/release checks pass.
