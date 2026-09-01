// Microduck beak liner reference — dimensions in millimetres.
//
// This is the physical counterpart of the simulator's shallow 24 mm-ball
// pocket.  It is intentionally parametric because Microduck beak revisions and
// printer shrinkage vary.  Measure the robot before printing; see README.md.

part = "both";              // "upper", "lower", or "both"
ball_diameter = 24.0;
inner_width = 28.0;         // clear space between side cheeks
cavity_depth = 26.0;        // rear stop to front lip
pad_thickness = 1.5;
wall_thickness = 2.0;
lip_radius = 2.0;
cheek_height = 5.0;
rear_stop_height = 5.0;
part_spacing = 10.0;
$fn = 64;

outer_width = inner_width + 2 * wall_thickness;

module transverse_rail(x, radius) {
    translate([x, outer_width, pad_thickness + radius])
        rotate([90, 0, 0])
            cylinder(h=outer_width, r=radius);
}

module lower_liner() {
    union() {
        cube([cavity_depth, outer_width, pad_thickness]);
        transverse_rail(cavity_depth, lip_radius);
    }
}

module upper_liner() {
    union() {
        cube([cavity_depth, outer_width, pad_thickness]);
        transverse_rail(cavity_depth, lip_radius);
        cube([wall_thickness, outer_width,
              pad_thickness + rear_stop_height]);
        cube([cavity_depth, wall_thickness,
              pad_thickness + cheek_height]);
        translate([0, outer_width - wall_thickness, 0])
            cube([cavity_depth, wall_thickness,
                  pad_thickness + cheek_height]);
    }
}

if (part == "upper") {
    upper_liner();
} else if (part == "lower") {
    lower_liner();
} else {
    lower_liner();
    translate([0, outer_width + part_spacing, 0]) upper_liner();
}
