set tcldir [file dirname [info script]]
source [file join $tcldir project.tcl]

# Real Vivado (out-of-context) synthesis of the standalone dot-product front-end
# (Part B, top = np_dots_only). Run AFTER build_dots_only.tcl has produced the
# RTL (csim+csynth) — it reads np_dots_only_prj/solution/syn/vhdl. csynth's DSP
# estimate becomes unreliable once the dot multipliers drop below the DSP48
# inference threshold (~10-bit operands, i.e. pmu <= 10), so this netlist-level
# pass is the trustworthy DSP/LUT/FF number for the sub-threshold pmu widths.
#
# The HLS project dir is always np_dots_only_prj (build_dots_only.tcl uses
# reset=1), so pass a report suffix to keep per-pmu runs from clobbering each
# other, e.g.
#   vivado -mode batch -source vivado_synth_dots.tcl -tclargs _pmu12
# writes vivado_synth_dots_pmu12.rpt. Part/clock come from project.tcl, so this
# matches whatever device build_dots_only.tcl synthesized against.
set suffix ""
if {$argc > 0} { set suffix [lindex $argv 0] }

add_files np_dots_only_prj/solution/syn/vhdl
synth_design -top np_dots_only -part $part
report_utilization -file vivado_synth_dots${suffix}.rpt
