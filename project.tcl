variable project_name
set project_name "nPELICAN"
variable backend
set backend "vivado"
variable part
# D2 comparison part: speed grade -2 to match the DeepSet (l1-jet-id) synthesis
# (xcvu13p-flga2577-2-e). Resource counts are grade-independent; -2 only changes timing
# closure. Revert to -1 below for the standalone nanoPELICAN baseline.
set part "xcvu13p-flga2577-2-e"
# Licensed-part override for Vivado synthesis: the archived vsynth runs used the U250
# (xcu250-figd2104-2L-e, same VU13P silicon: identical LUT/FF/DSP/BRAM/URAM totals, so
# counts are directly comparable; docs/resource_log.md "Vivado post-synthesis"). Set
#   export NPELICAN_PART=xcu250-figd2104-2L-e
# before vivado_synth.tcl (or build_prj.tcl) instead of editing this file on the box.
if {[info exists ::env(NPELICAN_PART)] && $::env(NPELICAN_PART) ne ""} {
    set part $::env(NPELICAN_PART)
    puts "INFO: part overridden by NPELICAN_PART = $part"
}
#set part "xcvu13p-flga2577-1-e"
#set part "xcvc1902-vsva2197-2MP-e-S"
variable clock_period
set clock_period 5
variable clock_uncertainty
set clock_uncertainty 12.5%
