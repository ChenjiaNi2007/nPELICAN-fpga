set tcldir [file dirname [info script]]
source [file join $tcldir project.tcl]

# Optional project-dir suffix for clock-sweep builds, e.g.
#   vivado -mode batch -source vivado_synth_split.tcl -tclargs _4ns
# reads nPELICAN_split_prj_4ns and writes vivado_synth_split_4ns.rpt.
set suffix ""
if {$argc > 0} { set suffix [lindex $argv 0] }

add_files ${project_name}_split_prj${suffix}/solution/syn/vhdl
synth_design -top ${project_name} -part $part
report_utilization -file vivado_synth_split${suffix}.rpt

# Post-opt numbers (same step as vivado_synth.tcl): like-for-like with the hls4ml MLP-Mixer flow.
opt_design -retarget -propconst -sweep -bram_power_opt -shift_register_opt
report_utilization -file vivado_synth_split_opt${suffix}.rpt
report_utilization -hierarchical -file vivado_synth_split_opt_hier${suffix}.rpt
