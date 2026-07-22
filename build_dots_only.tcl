# Standalone synthesis of the dot-product front-end (Part B).
# Reports the resources of JUST the dot portion (top = np_dots_only): no BN,
# aggregation, dense, or weights. Own project dir (np_dots_only_prj) so the
# monolith's / split's reports are never touched. csim + csynth only.
#
#   vitis_hls -f build_dots_only.tcl                 # runtime beam ports
#   vitis_hls -f build_dots_only.tcl const_beams=1   # Lever 5: hardwired beams
#
# Part/clock come from project.tcl (same device as the full-model runs, so the
# DSP/LUT/FF numbers are directly comparable to the monolith and split np_dots).
array set opt { reset 1 csim 1 synth 1 const_beams 0 }

set tcldir [file dirname [info script]]
source [file join $tcldir project.tcl]

foreach arg $::argv {
    foreach o [lsort [array names opt]] {
        regexp "(^|\\s)${o}=(\\w+)" $arg -> _ opt($o)
    }
}

set fw_cflags "-std=c++0x"
if {$opt(const_beams)} { append fw_cflags " -DNPELICAN_CONST_BEAMS" }

set prj_dir np_dots_only_prj
if {$opt(reset)} { open_project -reset $prj_dir } else { open_project $prj_dir }
set_top np_dots_only
add_files firmware/np_dots_only.cpp -cflags $fw_cflags
add_files -tb np_dots_only_tb.cpp   -cflags "-std=c++0x"
add_files -tb firmware/weights
if {$opt(reset)} { open_solution -reset "solution" } else { open_solution "solution" }
config_compile -name_max_length 80
set_part $part
config_schedule -enable_dsp_full_reg=false
create_clock -period $clock_period -name default
set_clock_uncertainty $clock_uncertainty default

if {$opt(csim)} {
    puts "***** C SIMULATION *****"
    csim_design
}
if {$opt(synth)} {
    puts "***** C/RTL SYNTHESIS *****"
    csynth_design
}
exit
