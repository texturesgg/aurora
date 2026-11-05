add_library(aurora_mtx STATIC
  lib/dolphin/mtx/mtx.c
  lib/dolphin/mtx/mtxstack.c
  lib/dolphin/mtx/mtxvec.c
  lib/dolphin/mtx/psmtx.c
  lib/dolphin/mtx/mtx44.c
  lib/dolphin/mtx/vec.c
  lib/dolphin/mtx/quat.c
)
add_library(aurora::mtx ALIAS aurora_mtx)
set_target_properties(aurora_mtx PROPERTIES FOLDER "aurora")

target_include_directories(aurora_mtx PUBLIC include)

# The paired-single routines fuse exactly where the Gekko does (psmtx.c), so the compiler
# must fuse nothing on its own
target_compile_options(aurora_mtx PRIVATE $<$<C_COMPILER_ID:GNU,Clang,AppleClang>:-ffp-contract=off>)
