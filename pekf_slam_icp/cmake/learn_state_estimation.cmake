# The state-estimation library (pose algebra, motion models, ICP, chi2) from learn-state-estimation, pinned to a
# commit. Only the C++ library target `state_estimation` is built: no tests, examples or Python module.
#
# Offline or with a local checkout:  colcon build --cmake-args -DFETCHCONTENT_SOURCE_DIR_LEARN_STATE_ESTIMATION=/path
include(FetchContent)
set(SE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SE_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    learn_state_estimation
    GIT_REPOSITORY https://github.com/Rudip1/learn-state-estimation.git
    GIT_TAG 2b41b6e8c1671ecfd6eaa5549ade59e3212c9019
    EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(learn_state_estimation)
