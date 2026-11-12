cmake_minimum_required(VERSION 3.16)
project(dasheng-audiogen LANGUAGES C CXX)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
add_compile_definitions(_USE_MATH_DEFINES)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(DASHENG_AUDIOGEN_METAL "Build with ggml-metal backend" ON)
option(DASHENG_AUDIOGEN_CUDA  "Build with ggml-cuda backend"  OFF)

if (APPLE AND DASHENG_AUDIOGEN_METAL)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()
if (DASHENG_AUDIOGEN_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()

add_subdirectory(third_party/ggml)

# Use the VCPKG_ROOT environment variable instead of a hardcoded path
if(NOT DEFINED ENV{VCPKG_ROOT})
    message(FATAL_ERROR "VCPKG_ROOT environment variable is not set. Please install vcpkg and set the environment variable.")
endif()
set(SENTENCEPIECE_DIR "$ENV{VCPKG_ROOT}/installed/x64-windows-static")
link_directories("${SENTENCEPIECE_DIR}/lib")
include_directories("${SENTENCEPIECE_DIR}/include")

add_library(dasheng_audiogen_core STATIC
    src/backend.cpp
    src/t5_encoder.cpp
    src/tokenizer.cpp
    src/content_adapter.cpp
    src/dit.cpp
    src/scheduler.cpp
    src/vocoder.cpp
    src/pipeline.cpp
)

target_include_directories(dasheng_audiogen_core PUBLIC
    src
    third_party/pocketfft
)

set(ABSL_LIBS
    absl_base.lib absl_borrowed_fixup_buffer.lib absl_city.lib absl_civil_time.lib
    absl_cord.lib absl_cordz_functions.lib absl_cordz_handle.lib absl_cordz_info.lib
    absl_cordz_sample_token.lib absl_cord_internal.lib absl_crc32c.lib absl_crc_cord_state.lib
    absl_crc_cpu_detect.lib absl_crc_internal.lib absl_debugging_internal.lib
    absl_decode_rust_punycode.lib absl_demangle_internal.lib absl_demangle_rust.lib
    absl_die_if_null.lib absl_examine_stack.lib absl_exponential_biased.lib
    absl_failure_signal_handler.lib absl_flags_commandlineflag.lib
    absl_flags_commandlineflag_internal.lib absl_flags_config.lib absl_flags_internal.lib
    absl_flags_marshalling.lib absl_flags_parse.lib absl_flags_private_handle_accessor.lib
    absl_flags_program_name.lib absl_flags_reflection.lib absl_flags_usage.lib
    absl_flags_usage_internal.lib absl_generic_printer_internal.lib absl_graphcycles_internal.lib
    absl_hash.lib absl_hashtablez_sampler.lib absl_hashtable_profiler.lib absl_int128.lib
    absl_kernel_timeout_internal.lib absl_leak_check.lib absl_log_entry.lib absl_log_flags.lib
    absl_log_globals.lib absl_log_initialize.lib absl_log_internal_check_op.lib
    absl_log_internal_conditions.lib absl_log_internal_fnmatch.lib absl_log_internal_format.lib
    absl_log_internal_globals.lib absl_log_internal_log_sink_set.lib absl_log_internal_message.lib
    absl_log_internal_nullguard.lib absl_log_internal_proto.lib absl_log_internal_structured_proto.lib
    absl_log_severity.lib absl_log_sink.lib absl_malloc_internal.lib absl_periodic_sampler.lib
    absl_poison.lib absl_profile_builder.lib absl_random_distributions.lib
    absl_random_internal_distribution_test_util.lib absl_random_internal_entropy_pool.lib
    absl_random_internal_platform.lib absl_random_internal_randen.lib
    absl_random_internal_randen_hwaes.lib absl_random_internal_randen_hwaes_impl.lib
    absl_random_internal_randen_slow.lib absl_random_internal_seed_material.lib
    absl_random_seed_gen_exception.lib absl_random_seed_sequences.lib absl_raw_hash_set.lib
    absl_raw_logging_internal.lib absl_scoped_set_env.lib absl_spinlock_wait.lib
    absl_stacktrace.lib absl_status.lib absl_statusor.lib absl_strerror.lib absl_strings.lib
    absl_strings_internal.lib absl_str_format_internal.lib absl_symbolize.lib
    absl_synchronization.lib absl_throw_delegate.lib absl_time.lib absl_time_zone.lib
    absl_tracing_internal.lib absl_utf8_for_code_point.lib absl_vlog_config_internal.lib
)

target_link_libraries(dasheng_audiogen_core PUBLIC
    ggml
    sentencepiece.lib
    libprotobuf.lib
    utf8_validity.lib
    utf8_range.lib
    ${ABSL_LIBS}
)

add_executable(dasheng-audiogen src/main.cpp)
target_link_libraries(dasheng-audiogen PRIVATE dasheng_audiogen_core)