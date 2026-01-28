# Encode CUDA kernel files into a C++ source file

# Read all header files from include/ directory in dependency order
# Base headers first, then headers that depend on them
SET(HEADER_ORDER
    "InterpolationBasis.cuh"
    "HermiteBasis.cuh"
    "TricubicCoefficients.cuh"
    "TriquinticCoefficients.cuh"
    "InvPowerChainRule.cuh"
    "TanhChainRule.cuh"
    "LJAnalyticalDerivatives.cuh"
    "GridInterpolation.cuh"
)

SET(HEADER_SOURCE "")
FOREACH(header_name ${HEADER_ORDER})
    SET(header "${KERNELS_DIR}/include/${header_name}")
    IF(EXISTS ${header})
        FILE(READ ${header} header_content)
        # Remove #include directives for local headers (they'll be inlined)
        STRING(REGEX REPLACE "#include \"[^\"]+\\.cuh\"\n" "" header_content "${header_content}")
        SET(HEADER_SOURCE "${HEADER_SOURCE}${header_content}\n")
    ENDIF()
ENDFOREACH()

# Read the kernel files and prepend headers
SET(SOURCE_CODE "${HEADER_SOURCE}\n")
# Convert space-separated string to list
STRING(REPLACE " " ";" KERNEL_FILE_LIST "${KERNEL_FILES}")
FOREACH(file ${KERNEL_FILE_LIST})
    FILE(READ ${file} file_source)
    # Remove #include directives for our local headers since we're inlining them
    STRING(REGEX REPLACE "#include \"include/[^\"]+\"\n" "" file_source "${file_source}")
    SET(SOURCE_CODE "${SOURCE_CODE}${file_source}")
ENDFOREACH()

# Generate the cpp file using raw string literal with custom delimiter
FILE(WRITE ${KERNELS_CPP}
"#include \"CudaGridForceKernelSources.h\"

namespace GridForcePlugin {

std::string CudaGridForceKernelSources::gridForceKernel = R\"KERNELSRC(
${SOURCE_CODE}
)KERNELSRC\";

} // namespace GridForcePlugin
")
