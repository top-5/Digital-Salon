set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
find_package(CUDAToolkit 12.8 REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(nlohmann_json REQUIRED)
find_package(OpenGL REQUIRED COMPONENTS OpenGL EGL)
find_package(glm REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(assimp REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options system filesystem)
find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter Development)
find_package(OpenMP REQUIRED)
find_library(FREEIMAGE_LIBRARY freeimage REQUIRED)

# Keep the working docking-enabled ImGui API shipped with the viewer.
set(IMGUI_DIR "${PROJECT_SOURCE_DIR}/extern/include/imgui")
add_library(imgui STATIC
    ${IMGUI_DIR}/imgui.cpp ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp ${IMGUI_DIR}/imgui_stdlib.cpp
    ${IMGUI_DIR}/imgui_impl_glfw.cpp ${IMGUI_DIR}/imgui_impl_opengl3.cpp)
target_link_libraries(imgui PUBLIC glfw OpenGL::GL)

file(GLOB_RECURSE CPP_SOURCES CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/*.cpp" "${PROJECT_SOURCE_DIR}/extern/hairsim/src/*.cpp")
file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/extern/hairsim/src/*.cu")
add_executable(${EXEC_NAME} ${CPP_SOURCES} ${CUDA_SOURCES})
# System dependencies must precede the legacy bundled assimp/glm headers.
target_include_directories(${EXEC_NAME} BEFORE PRIVATE
    ${Boost_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
target_include_directories(${EXEC_NAME} PRIVATE
    "${PROJECT_SOURCE_DIR}/extern/hairsim/include"
    "${PROJECT_SOURCE_DIR}/extern/hairsim/include/cuda"
    "${PROJECT_SOURCE_DIR}/extern/hairsim/src"
    "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/extern")
# Only ImGui needs the bundled include root: expose it without shadowing system headers.
target_include_directories(${EXEC_NAME} PRIVATE "${PROJECT_BINARY_DIR}/compat")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/compat")
file(CREATE_LINK "${IMGUI_DIR}" "${PROJECT_BINARY_DIR}/compat/imgui" SYMBOLIC)
set_target_properties(${EXEC_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
    CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(${EXEC_NAME} PRIVATE
    CUDA::cudart Eigen3::Eigen nlohmann_json::nlohmann_json glm::glm
    Python3::Python GLEW::GLEW glfw GLUT::GLUT OpenGL::GL OpenGL::GLU OpenGL::EGL
    ${FREEIMAGE_LIBRARY} assimp::assimp imgui
    Boost::program_options Boost::system Boost::filesystem OpenMP::OpenMP_CXX)
