#ifdef JL_CODEGEN_FALLBACKS_STATIC
#undef _GNU_SOURCE
#include "../cli/loader.h"
#include "../cli/jl_exports.h"

void *jl_method_table;

JL_DLLEXPORT const char *jl_get_libdir(void)
{
  return "TODO";
}

#endif
