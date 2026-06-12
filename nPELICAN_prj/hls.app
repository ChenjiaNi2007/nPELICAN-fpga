<AutoPilot:project xmlns:AutoPilot="com.autoesl.autopilot.project" top="nPELICAN" name="nPELICAN_prj">
    <files>
        <file name="../../tb_data" sc="0" tb="1" cflags="-Wno-unknown-pragmas" csimflags="" blackbox="false"/>
        <file name="../../firmware/weights" sc="0" tb="1" cflags="-Wno-unknown-pragmas" csimflags="" blackbox="false"/>
        <file name="../../nPELICAN_tb.cpp" sc="0" tb="1" cflags="-std=c++0x -Wno-unknown-pragmas" csimflags="" blackbox="false"/>
        <file name="firmware/nPELICAN.cpp" sc="0" tb="false" cflags="-std=c++0x" csimflags="" blackbox="false"/>
    </files>
    <solutions>
        <solution name="solution" status=""/>
    </solutions>
    <Simulation argv="">
        <SimFlow name="csim" setup="false" optimizeCompile="false" clean="false" ldflags="" mflags=""/>
    </Simulation>
</AutoPilot:project>

