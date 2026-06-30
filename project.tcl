variable project_name
set project_name "nPELICAN"
variable backend
set backend "vivado"
variable part
# D2 comparison part: speed grade -2 to match the DeepSet (l1-jet-id) synthesis
# (xcvu13p-flga2577-2-e). Resource counts are grade-independent; -2 only changes timing
# closure. Revert to -1 below for the standalone nanoPELICAN baseline.
set part "xcvu13p-flga2577-2-e"
#set part "xcvu13p-flga2577-1-e"
#set part "xcvc1902-vsva2197-2MP-e-S"
variable clock_period
set clock_period 5
variable clock_uncertainty
set clock_uncertainty 12.5%
