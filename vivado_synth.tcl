set tcldir [file dirname [info script]]
source [file join $tcldir project.tcl]

# Optional project-dir suffix for clock-sweep builds, e.g.
#   vivado -mode batch -source vivado_synth.tcl -tclargs _4ns
# reads nPELICAN_prj_4ns and writes vivado_synth_4ns.rpt.
set suffix ""
if {$argc > 0} { set suffix [lindex $argv 0] }

add_files ${project_name}_prj${suffix}/solution/syn/vhdl
synth_design -top ${project_name} -part $part
report_utilization -file vivado_synth${suffix}.rpt

# Post-opt numbers for the MLP-Mixer comparison: the hls4ml/Mixer flow reports utilization after
# opt_design, so the like-for-like nanoPELICAN figure is vivado_synth_opt.rpt (post-synth-only
# vivado_synth.rpt stays for continuity with every earlier nanoPELICAN report).
opt_design -retarget -propconst -sweep -bram_power_opt -shift_register_opt
report_utilization -file vivado_synth_opt${suffix}.rpt
report_utilization -hierarchical -file vivado_synth_opt_hier${suffix}.rpt
