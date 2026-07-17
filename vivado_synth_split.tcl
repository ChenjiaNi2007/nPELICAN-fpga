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
