# Try to find cnpy
#
# Inputs:
#   SNDFILE_INC_DIR: include directory for sndfile headers
#   SNDFILE_LIB_DIR: directory containing sndfile libraries
#   SNDFILE_ROOT_DIR: directory containing sndfile installation
#
# Defines:
#  SNDFILE_FOUND - system has libsndfile
#  SNDFILE_INCLUDE_DIRS - the libsndfile include directory
#  SNDFILE_LIBRARIES - Link these to use libsndfile
#
set(INC_DIR /usr/local/include)
set(LINK_DIR /usr/local/lib)

find_path(
  CNPY_INCLUDE_DIR
    cnpy.h
  PATHS
  ${INC_DIR}
  )

find_library(
  CNPY_LIBRARY
  cnpy
  PATHS
  ${LINK_DIR}
  )

set(CNPY_INCLUDE_DIRS
  ${CNPY_INCLUDE_DIR}
  )
set(CNPY_LIBRARIES
  ${CNPY_LIBRARY}
  )

mark_as_advanced(CNPY_INCLUDE_DIRS CNPY_LIBRARIES)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CNPY DEFAULT_MSG CNPY_INCLUDE_DIRS CNPY_LIBRARIES)

if (CNPY_FOUND)
  message(STATUS "Found cnpy: (lib: ${CNPY_LIBRARIES} include: ${CNPY_INCLUDE_DIRS}")
else()
  message(STATUS "cnpy not found.")
endif()
