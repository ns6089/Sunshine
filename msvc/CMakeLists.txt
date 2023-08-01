cmake_minimum_required(VERSION 3.21)
project(sunshine)
set(CMAKE_CXX_STANDARD 20)

add_compile_definitions(NOMINMAX _WIN32_WINNT=0x0A00 WINVER=0x0A00 SUNSHINE_ASSETS_DIR="assets" SUNSHINE_PLATFORM="windows")
add_compile_options(/utf-8 /wd4828)

# boost (vcpkg), adds vcpkg include and library directories
find_package(Boost 1.82.0 COMPONENTS locale log filesystem program_options json REQUIRED)
include_directories($<IF:$<CONFIG:Debug>,${Boost_INCLUDE_DIR_DEBUG},${Boost_INCLUDE_DIR_RELEASE}>)
link_directories($<IF:$<CONFIG:Debug>,${Boost_LIBRARY_DIR_DEBUG},${Boost_LIBRARY_DIR_RELEASE}>)
list(APPEND SUNSHINE_LIBRARIES ${Boost_LIBRARIES})

# openssl (vcpkg)
find_package(OpenSSL REQUIRED)
list(APPEND SUNSHINE_LIBRARIES OpenSSL::SSL OpenSSL::Crypto)

# curl (vcpkg)
find_package(CURL CONFIG REQUIRED)
list(APPEND SUNSHINE_LIBRARIES ${CURL_LIBRARIES})

# ffmpeg (vcpkg)
list(APPEND SUNSHINE_LIBRARIES avcodec avutil swscale swresample libmfx libx264)

# miniupnpc (vcpkg)
add_compile_definitions(MINIUPNP_STATICLIB)
list(APPEND SUNSHINE_LIBRARIES miniupnpc)

# opus (vcpkg)
find_package(OPUS CONFIG REQUIRED)
list(APPEND SUNSHINE_LIBRARIES ${OPUS_LIBRARIES})

# cuda (system)
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})
list(APPEND SUNSHINE_LIBRARIES CUDA::cuda_driver CUDA::cudart_static)

# vigemclient (third-party)
list(APPEND VIGEM_SOURCES ../third-party/ViGEmClient/src/ViGEmClient.cpp)
include_directories(../third-party/ViGEmClient/include)

# nanors (third-party)
list(APPEND NANORS_SOURCES ../third-party/nanors/rs.c)
set_source_files_properties(../third-party/nanors/rs.c PROPERTIES COMPILE_FLAGS "/FI deps/obl/autoshim.h") # /Qvec-report:2
include_directories(../third-party/nanors)
include_directories(../third-party/nanors/deps/obl)

# moonlight-common-c (third-party)
add_subdirectory(../third-party/moonlight-common-c/enet enet)
include_directories(../third-party/moonlight-common-c/enet/include)
list(APPEND SUNSHINE_LIBRARIES enet)
list(APPEND MOONLIGHT_SOURCES ../third-party/moonlight-common-c/src/RtspParser.c)

list(APPEND SUNSHINE_LIBRARIES DXGI D3D11 Dwmapi D3DCompiler Iphlpapi Userenv Wtsapi32 Setupapi Synchronization Mfuuid Strmiids)

include_directories(.)
include_directories(..)
include_directories(../third-party)
include_directories(../third-party/ffmpeg-windows-x86_64/include)
include_directories(../third-party/nvapi-open-source-sdk)
include_directories(BEFORE ../third-party/nv-codec-headers/include)

file(GLOB_RECURSE SUNSHINE_SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/*.h")
list(FILTER SUNSHINE_SOURCES EXCLUDE REGEX "src/platform/linux/.*" )
list(FILTER SUNSHINE_SOURCES EXCLUDE REGEX "src/platform/macos/.*" )

add_executable(sunshine ${SUNSHINE_SOURCES} ${VIGEM_SOURCES} ${NANORS_SOURCES} ${MOONLIGHT_SOURCES})
target_link_libraries(sunshine PRIVATE ${SUNSHINE_LIBRARIES})
