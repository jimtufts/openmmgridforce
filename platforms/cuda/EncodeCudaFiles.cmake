# Encode CUDA kernel files into a C++ source file

# Read all header files from include/ directory
FILE(GLOB HEADER_FILES "${KERNELS_DIR}/include/*.cuh")
SET(HEADER_SOURCE "")
FOREACH(header ${HEADER_FILES})
    FILE(READ ${header} header_content)
    SET(HEADER_SOURCE "${HEADER_SOURCE}${header_content}\n")
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
