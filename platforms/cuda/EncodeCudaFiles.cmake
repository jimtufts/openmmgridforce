# Encode CUDA kernel files into a C++ source file

# Read the kernel files
SET(SOURCE_CODE "")
FOREACH(file ${KERNEL_FILES})
    FILE(READ ${file} file_source)
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
